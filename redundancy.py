"""关键路由双通道规划（critical-route redundant planning）。

用户勾选若干**必须抗单点断路**的关键路由，并提交**最多允许新增的跳线数**。
系统沿现有设备、插孔对与全常态 / 半常态 / 直通语义，为**全部路由**联合选择
一套**可同时安装**的通路方案：

  - 每条关键路由得到两条通路——**主通路（primary）**与**备用通路（backup）**；
  - 非关键路由仍只需一条在最终占用下**保持可用**的通路；
  - 所有通路的跳线在前面板上**同时安装**（同一插孔物理上只能插一根跳线，
    全局插孔占用唯一），先求「全部路由可满足」，再求**跳线总数最少**
    （相对输入顺序贪心基线的新增跳线数不超过用户预算）。

抗单点断路判据（逐条断点做物理仿真，而非仅做静态去重）：

  枚举方案中每一根**已安装跳线被拔断**与每一条**最终状态下生效的常态被
  切断**，在该故障快照上重新按最终插头占用推导常态活性（拔跳线会释放插孔，
  全 / 半常态可能因此重新接通），要求关键路由的主、备两条链路中至少一条仍然
  逐段连通。由此自然推出：

    1. 主备两条通路不得共用跳线（共用则拔一根同归于尽；插孔唯一使不同通路
       物理上也不可能插出同一根跳线）；
    2. 主备两条通路不得共用常态连接（共用则切断该常态同归于尽）——搜索时
       直接作为候选过滤；
    3. **半常态上插孔并送按最终占用判定**：上插孔插线时常态保持（信号并送），
       走常态的通路与用上插孔出跳线的通路可同时存活；但下插孔插线会断开常态，
       走该常态的通路转为**热备**——跳线被拔后常态自动恢复，切断常态则跳线
       仍在，两者恰好互为备份。全常态任一侧插线同理转为热备，但是否真能在
       拔线后恢复还要看其余插头（由故障仿真精确判定，例如第三方路由仍插着
       该对的上插孔时，全常态不会因拔下备线而恢复）。

  设备内部连接不出现在单点故障模型里（不会被拔断 / 切断），故主备在设备
  内部汇合是允许的（典型：两条通路在目标设备输入端经「常态 / 跳线」热备
  汇合）。

搜索策略（确定性，与 isolation 同构）：

  - 槽位：每条关键路由两个槽位（主先备后），每条非关键路由一个槽位；
  - MRV（最少候选优先）选槽 + 候选路径按跳线数升序枚举（环无关有界枚举），
    备用候选额外过滤掉与主路共用常态的路径；
  - 迭代加深跳线总预算（0 … 基线跳数 + maxNewCables），首个可行方案即
    跳线最少方案；
  - 剪枝：某槽位在当前插头下已无候选（占用单调增长）、剩余槽位最短跳线数
    之和的下界超预算；
  - 全部槽位落位后做最终占用校验与逐断点仿真。

无解时抛 :class:`RedundancyConflict`（409），携带：关键路由被迫共用的
**冲突断点**（共用常态 / 争用独占插孔，附备用理想路径见证）、终止状态的
**占用插孔 / 跳线**、**未满足路由**与贪心基线拓扑——**不返回接线步骤，
不提交部分方案**。
"""

from collections import defaultdict

from solver import (MODE_LABELS, ConfigError, Solver,
                    detect_loops, diagnose, pairs_summary, parse_config,
                    plug_note, _pid)
from isolation import (Node, bind, enumerate_paths, greedy_baseline,
                       active_normal_ids, _edge_key)

PLAN_NODE_CAP = 120_000        # 回溯搜索节点上限（超出返回 inconclusive）
MAX_CABLE_DEPTH = 4            # 单条通路候选最多经过的跳线数


class RedundancyError(ConfigError):
    """双通道请求非法（400 REDUNDANCY_INVALID）。"""

    def __init__(self, message, details=None):
        super().__init__(message, code="REDUNDANCY_INVALID",
                         details=details or {})


class RedundancyConflict(ConfigError):
    """不存在满足关键路由双通道约束的完整规划（409 REDUNDANCY_CONFLICT）。"""

    def __init__(self, message, payload):
        super().__init__(message, code="REDUNDANCY_CONFLICT", details=payload)
        self.payload = payload


# ---- 请求解析 -------------------------------------------------------------


def parse_redundancy(raw, n_routes):
    """校验 {criticalRoutes, maxNewCables}，返回 (critical 序号升序列表, 预算)。"""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise RedundancyError(
            '请求必须是 JSON 对象，形如 {"criticalRoutes": [0, 2], '
            '"maxNewCables": 3}')
    crit_raw = raw.get("criticalRoutes")
    if not isinstance(crit_raw, list, ):
        raise RedundancyError("criticalRoutes 必须是路由序号数组（从 0 开始）")
    if not crit_raw:
        raise RedundancyError("请至少勾选一条必须抗单点断路的关键路由")
    critical, seen = [], set()
    for x in crit_raw:
        if isinstance(x, bool) or not isinstance(x, int):
            raise RedundancyError(f"关键路由序号必须是整数，得到 {x!r}")
        if x < 0 or x >= n_routes:
            raise RedundancyError(
                f"关键路由序号 {x} 越界：当前共 {n_routes} 条路由（序号 0…"
                f"{n_routes - 1}）")
        if x in seen:
            raise RedundancyError(f"关键路由序号 {x} 重复勾选")
        seen.add(x)
        critical.append(x)
    critical.sort()

    budget_raw = raw.get("maxNewCables")
    if isinstance(budget_raw, bool) or not isinstance(budget_raw, int):
        raise RedundancyError("maxNewCables 必须是非负整数（最多允许新增的跳线数）")
    if budget_raw < 0:
        raise RedundancyError("maxNewCables 不能为负数")
    if budget_raw > 10_000:
        raise RedundancyError("maxNewCables 过大（上限 10000）")
    return critical, budget_raw


# ---- 提交（不保护常态：双通道允许跨路径以热备 / 并送方式改变常态活性） -------


def commit_red(solver, node, items):
    """按最终插头占用提交一条通路。

    与 ``solver.Solver.commit`` / ``isolation.commit`` 的差别：**不把通路
    经过的常态登记为受保护常态**——双通道规划里后续跳线允许以半常态并送或
    主备热备的方式改变这些常态的活性，常态是否生效一律由最终插头占用决定。
    返回 (新 Node, {(fj,tj): cid})。
    """
    plugs = dict(node.plugs)
    cables = [dict(c) for c in node.cables]
    added = 0
    cid_map = {}
    for kind, payload in items:
        if kind == "edge" and payload[0] == "cable":
            _, fj, tj = payload
            if fj not in plugs:
                cid = node.next_cid + added
                added += 1
                cables.append({
                    "id": cid, "fromJack": fj, "toJack": tj,
                    "fromPort": solver.jack_port[fj],
                    "toPort": solver.jack_port[tj],
                })
                plugs[fj] = cid
                plugs[tj] = cid
            cid_map[(fj, tj)] = plugs[fj]
    new = Node(plugs, cables, set(), node.next_cid + added,
               dict(node.commits), list(node.order))
    return new, cid_map


# ---- 通路资源 -------------------------------------------------------------


def path_normals(items):
    return {payload[1]["id"] for kind, payload in items
            if kind == "edge" and payload[0] == "normal"}


def path_cables(items):
    return [(payload[1], payload[2]) for kind, payload in items
            if kind == "edge" and payload[0] == "cable"]


def path_sig(items):
    return tuple(_edge_key(payload) for kind, payload in items
                 if kind == "edge")


# ---- 最终占用 / 故障仿真 ---------------------------------------------------


def active_normals_under(solver, pairs, plugs, broken=()):
    """给定插头占用与强制切断集合，返回生效常态编号集合。"""
    ids = active_normal_ids(solver, pairs, plugs)
    return ids - set(broken)


def cable_map(cables):
    return {(c["fromJack"], c["toJack"]): c["id"] for c in cables}


def chain_alive(items, cmap, active_nids, broken_cables=()):
    """逐段判定一条已安装链路在故障快照下是否连通。

    cmap:          {(fromJack,toJack): cableId}（快照中仍在役的跳线）；
    active_nids:   快照中生效的常态编号集合；
    broken_cables: 被拔断跳线的 {(fj,tj)} 集合（也可直接不在 cmap 中）。
    """
    broken_cables = set(broken_cables)
    for kind, payload in items:
        if kind != "edge":
            continue
        if payload[0] == "device":
            continue
        if payload[0] == "normal":
            if payload[1]["id"] not in active_nids:
                return False
        else:
            key = (payload[1], payload[2])
            if key in broken_cables or key not in cmap:
                return False
    return True


def simulate_breaks(solver, pairs, cables, routes_chain):
    """枚举全部单点断点并仿真每条路由的幸存情况。

    routes_chain: {idx: {"critical": bool, "slots": {role: items}}}，
                  非关键路由 role 为 "single"。
    返回 (breakpoints, preview)：
      breakpoints: [{type, id, label, affectedCount, affectedRoutes, ports}]
      preview:     同序，每项额外带 routes: [{index, critical, roles:{role:alive},
                    survivor, serviceLost}]
    """
    plugs = {}
    for c in cables:
        plugs[c["fromJack"]] = c["id"]
        plugs[c["toJack"]] = c["id"]
    live_normals = active_normals_under(solver, pairs, plugs)

    # 资源 -> 用到它的路由（断点排序用）
    cable_users, normal_users = defaultdict(set), defaultdict(set)
    for idx, info in routes_chain.items():
        for role, items in info["slots"].items():
            for kind, payload in items:
                if kind != "edge":
                    continue
                if payload[0] == "cable":
                    cable_users[(payload[1], payload[2])].add(idx)
                elif payload[0] == "normal":
                    normal_users[payload[1]["id"]].add(idx)

    points = []
    for c in cables:
        key = (c["fromJack"], c["toJack"])
        points.append({
            "type": "cable", "id": c["id"], "key": key,
            "label": f"跳线 #{c['id']}：{c['fromJack']}（{c['fromPort']}）"
                     f" ↔ {c['toJack']}（{c['toPort']}）",
            "users": sorted(cable_users.get(key, ())),
            "ports": [{"jack": c["fromJack"], "port": c["fromPort"]},
                      {"jack": c["toJack"], "port": c["toPort"]}],
        })
    for nid in sorted(live_normals):
        pair = solver.pair_of_jack[f"{nid}T"]
        points.append({
            "type": "normal", "id": nid, "key": ("normal", nid),
            "label": f"常态连接 第{nid}对（{MODE_LABELS[pair['mode']]}）："
                     f"{_pid(pair['top'])} → {_pid(pair['bottom'])}",
            "users": sorted(normal_users.get(nid, ())),
            "ports": [{"jack": f"{nid}T", "port": _pid(pair["top"])},
                      {"jack": f"{nid}B", "port": _pid(pair["bottom"])}],
        })

    preview = []
    base_cmap = cable_map(cables)
    for p in points:
        if p["type"] == "cable":
            fj, tj = p["key"]
            snap_plugs = {j: cid for j, cid in plugs.items()
                          if j not in (fj, tj)}
            snap_cmap = {k: cid for k, cid in base_cmap.items() if k != p["key"]}
            snap_normals = active_normals_under(solver, pairs, snap_plugs)
        else:
            snap_plugs = plugs
            snap_cmap = base_cmap
            snap_normals = live_normals - {p["id"]}

        rows, critical_lost, noncritical_lost = [], [], []
        for idx, info in routes_chain.items():
            roles_alive = {}
            for role, items in info["slots"].items():
                roles_alive[role] = chain_alive(items, snap_cmap, snap_normals)
            if info["critical"]:
                pa, ba = roles_alive.get("primary", False), \
                         roles_alive.get("backup", False)
                survivor = "primary" if pa else ("backup" if ba else "none")
                service_lost = not (pa or ba)
                if service_lost:
                    critical_lost.append(idx)
            else:
                survivor = "path" if roles_alive.get("single") else "none"
                service_lost = survivor == "none"
                if service_lost:
                    noncritical_lost.append(idx)
            rows.append({"index": idx, "critical": info["critical"],
                         "primaryAlive": roles_alive.get("primary"),
                         "backupAlive": roles_alive.get("backup"),
                         "pathAlive": roles_alive.get("single"),
                         "survivor": survivor, "serviceLost": service_lost})
        rows.sort(key=lambda r: r["index"])
        preview.append({
            "type": p["type"], "id": p["id"], "label": p["label"],
            "ports": p["ports"], "routes": rows,
            "affectedCount": len(p["users"]),
            "affectedRoutes": p["users"],
            "criticalLost": critical_lost,
            "nonCriticalLost": noncritical_lost,
        })

    preview.sort(key=lambda f: (-len(f["affectedRoutes"]),
                                0 if f["type"] == "cable" else 1, f["id"]))
    for f in preview:
        f.pop("affectedCount", None)
    return [{k: v for k, v in f.items()
             if k in ("type", "id", "label", "ports", "affectedCount",
                      "affectedRoutes")} for f in
            sorted(points, key=lambda p: (-len(p["users"]),
                                          0 if p["type"] == "cable" else 1,
                                          p["id"]))], preview


# ---- 主搜索 ---------------------------------------------------------------


def plan_redundancy(config, red_raw):
    """关键路由双通道规划。成功返回结果 dict；失败抛 RedundancyConflict。"""
    devices, pairs, routes, port_jack = parse_config(config)
    critical, max_new = parse_redundancy(red_raw, len(routes))
    critical_set = set(critical)
    solver = Solver(devices, pairs, routes, port_jack)

    # ---- 输入顺序贪心基线：基线必须本身可通；B = 基线跳线数 -------------
    greedy_node, greedy_failed = greedy_baseline(solver, routes)
    bind(solver, greedy_node)
    if greedy_failed or len(greedy_node.order) < len(routes):
        raise RedundancyConflict(
            "基线规划（输入顺序逐条最短路）本身存在无法路由的请求：必须先让"
            "全部路由可通，才能为关键路由规划主备双通道",
            _baseline_conflict_payload(solver, pairs, routes, greedy_node,
                                       greedy_failed, critical))
    baseline_cables = len(greedy_node.cables)
    baseline_conns = {(c["fromJack"], c["toJack"]) for c in greedy_node.cables}

    # ---- 槽位定义 --------------------------------------------------------
    slots = []          # [(routeIdx, role)]，role: primary/backup/single
    rinfo = []          # (idx, src, dst, critical)
    for idx, r in enumerate(routes):
        src, dst = _pid(r["source"]), _pid(r["target"])
        rinfo.append((idx, src, dst, idx in critical_set))
        if idx in critical_set:
            slots.append((idx, "primary"))
            slots.append((idx, "backup"))
        else:
            slots.append((idx, "single"))
    slot_of = {s: k for k, s in enumerate(slots)}

    def slot_src_dst(slot):
        idx, role = slot
        return rinfo[idx][1], rinfo[idx][2]

    stats = {"nodes": 0, "truncated": False, "deepSlots": set()}
    aborted = [False]
    best = {"done": -1, "node": None, "reasons": [], "missing": []}
    paths_memo = {}
    short_memo = {}

    def plugs_key(node):
        return frozenset(node.plugs)

    def primary_sig(node, idx):
        """备用槽位所需的主路签名（用于候选过滤与缓存键）。"""
        it = node.commits.get((idx, "primary"))
        return path_sig(it[0]) if it else None

    def slot_candidates(node, slot):
        """当前插头状态下该槽位的候选通路 [(items, cnum, normals(set))]。

        备用槽位额外要求：① 与主路不是同一条路径；② 不与主路共用常态
        （共用常态则切断时同归于尽，且约束单调，可直接过滤）。插孔争用由
        全局「一插孔一插头」在枚举/提交层保证；半常态并送 / 热备能否真正
        互为备份由落位后的逐断点仿真终判。
        """
        idx, role = slot
        src, dst = slot_src_dst(slot)
        pkey = (plugs_key(node), idx, role,
                primary_sig(node, idx) if role == "backup" else None,
                MAX_CABLE_DEPTH)
        if pkey in paths_memo:
            return paths_memo[pkey]
        bind(solver, node)
        paths, trunc = enumerate_paths(solver, src, dst, MAX_CABLE_DEPTH)
        if trunc:
            stats["truncated"] = True
        out = []
        p_items = None
        if role == "backup":
            p_commit = node.commits.get((idx, "primary"))
            p_items = p_commit[0] if p_commit else None
            p_normals = path_normals(p_items) if p_items else set()
        for items, cnum in paths:
            if p_items is not None:
                if path_sig(items) == path_sig(p_items):
                    continue
                if path_normals(items) & p_normals:
                    continue
            out.append((items, cnum, path_normals(items)))
        paths_memo[pkey] = out
        return out

    def shortest_cnum(node, slot):
        """当前插头状态下该槽位最短跳线数；无候选返回 None（带记忆）。"""
        key = (plugs_key(node), slot)
        if key not in short_memo:
            cands = slot_candidates(node, slot)
            short_memo[key] = min((c[1] for c in cands), default=None)
        return short_memo[key]

    def available_slots(node):
        done = set(node.order)
        out = []
        for slot in slots:
            idx, role = slot
            if slot in done:
                continue
            if role == "backup" and (idx, "primary") not in done:
                continue
            out.append(slot)
        return out

    def lower_bound(node, remaining):
        """可通行剪枝：任一剩余槽位当前已无候选则该分支不可行。

        注意不能用“各槽位最短跳线数之和”做预算下界——后落位的槽位可以
        复用先落位槽位安装的同一根跳线（跳线一旦在役，任何路由都能沿其
        from→to 边通过，不再占新插孔），求和会高估、误剪可行分支
        （与 isolation 搜索同一结论）。
        """
        for slot in remaining:
            if shortest_cnum(node, slot) is None:
                return None
        return node.cable_count

    def attach(node, slot, items, cid_map, normals):
        new = Node(node.plugs, node.cables, node.protected, node.next_cid,
                   dict(node.commits), node.order + [slot])
        new.commits[slot] = (items, cid_map, normals)
        return new

    def validate(node):
        """全部槽位落位后的最终占用 + 逐断点校验。返回失败原因列表（空=可行）。"""
        bind(solver, node)
        plugs = node.plugs
        live = active_normals_under(solver, pairs, plugs)
        cmap = cable_map(node.cables)

        chains = {}
        for idx, src, dst, crit in rinfo:
            if crit:
                chains[idx] = {"critical": True,
                               "slots": {"primary": node.commits[(idx, "primary")][0],
                                         "backup": node.commits[(idx, "backup")][0]}}
            else:
                chains[idx] = {"critical": False,
                               "slots": {"single": node.commits[(idx, "single")][0]}}

        reasons = []
        # ① 非关键路由：最终占用下必须保持可用
        for idx, src, dst, crit in rinfo:
            if crit:
                continue
            items = chains[idx]["slots"]["single"]
            if not chain_alive(items, cmap, live):
                reasons.append(("noncritical-silent", idx,
                                f"非关键路由 {idx + 1}（{src} → {dst}）在最终"
                                "插孔占用下不通声"))
        # ② 关键路由：无故障时至少一条通；每个单点断点后至少一条幸存
        for idx, src, dst, crit in rinfo:
            if not crit:
                continue
            pa0 = chain_alive(chains[idx]["slots"]["primary"], cmap, live)
            ba0 = chain_alive(chains[idx]["slots"]["backup"], cmap, live)
            if not (pa0 or ba0):
                reasons.append(("critical-silent", idx,
                                f"关键路由 {idx + 1}（{src} → {dst}）的主备通路"
                                "在最终插孔占用下均不通声"))
                continue
            for c in node.cables:
                snap_plugs = {j: cid for j, cid in plugs.items()
                              if j not in (c["fromJack"], c["toJack"])}
                snap_cmap = {k: v for k, v in cmap.items()
                             if k != (c["fromJack"], c["toJack"])}
                snap_live = active_normals_under(solver, pairs, snap_plugs)
                pa = chain_alive(chains[idx]["slots"]["primary"], snap_cmap, snap_live)
                ba = chain_alive(chains[idx]["slots"]["backup"], snap_cmap, snap_live)
                if not (pa or ba):
                    reasons.append(("critical-break", idx,
                                    f"关键路由 {idx + 1}（{src} → {dst}）在跳线 "
                                    f"#{c['id']} 被拔断后主备同时失效"))
                    break
            else:
                for nid in sorted(live):
                    if not (chain_alive(chains[idx]["slots"]["primary"], cmap,
                                        live - {nid}) or
                            chain_alive(chains[idx]["slots"]["backup"], cmap,
                                        live - {nid})):
                        reasons.append(("critical-break", idx,
                                        f"关键路由 {idx + 1}（{src} → {dst}）在第 "
                                        f"{nid} 对常态被切断后主备同时失效"))
                        break
        return reasons, chains

    abort = [False]

    def record_terminal(node, reasons, missing):
        if len(node.order) > best["done"]:
            best.update(done=len(node.order), node=node, reasons=reasons,
                        missing=list(missing))

    def dfs(node, budget):
        if abort[0]:
            return None
        stats["nodes"] += 1
        if stats["nodes"] > PLAN_NODE_CAP:
            abort[0] = True
            aborted[0] = True
            return None
        remaining = available_slots(node)
        if not remaining:
            reasons, chains = validate(node)
            if not reasons:
                return node, chains
            record_terminal(node, reasons, [])
            return None

        # 可通性 / 深度剪枝（占用单调，当前无候选则永无候选）
        for slot in remaining:
            cands = slot_candidates(node, slot)
            if not cands:
                record_terminal(node, [("no-candidate", slot[0], "")],
                                [s for s in slots if s not in node.order])
                return None
            if shortest_cnum(node, slot) > MAX_CABLE_DEPTH:
                stats["deepSlots"].add(slot)
        lb = lower_bound(node, remaining)
        if lb is None or lb > budget:
            record_terminal(node, [("lower-bound", None, "")],
                            [s for s in slots if s not in node.order])
            return None

        # MRV：候选最少的槽位先落位；同数时按路由序号、主先备后
        def mrv_key(slot):
            return (len(slot_candidates(node, slot)), slot[0],
                    0 if slot[1] in ("primary", "single") else 1)

        nxt = min(remaining, key=mrv_key)
        room = budget - node.cable_count
        for items, cnum, normals in slot_candidates(node, nxt):
            if cnum > room:
                continue
            child, cid_map = commit_red(solver, node, items)
            child = attach(child, nxt, items, cid_map, normals)
            found = dfs(child, budget)
            if found is not None or abort[0]:
                return found
        return None

    cap = min(baseline_cables + max_new, len(pairs))
    feasible = None
    for budget in range(0, cap + 1):
        res = dfs(Node(), budget)
        if res is not None or aborted[0]:
            feasible = res
            break

    note = None
    if aborted[0]:
        note = (f"双通道回溯搜索超过 {PLAN_NODE_CAP} 个节点上限：已列出搜索中"
                "发现的冲突断点，但未能在时限内证明不存在可行方案"
                "（inconclusive，不提交部分方案）")
    elif feasible is None and stats["deepSlots"]:
        note = (f"部分通路需要超过 {MAX_CABLE_DEPTH} 条跳线：超出候选枚举深度，"
                "结果可能非最优或存在未检出的可行规划（inconclusive）")

    if feasible is None:
        # 预算是否是唯一瓶颈：在不受预算约束下再搜一次（仅失败时付出该成本）
        budget_limited = False
        if not aborted[0] and not stats["deepSlots"]:
            hard_cap = len(pairs)
            if hard_cap > cap:
                # 预算探针：独立的节点计数 / 终止记录 / 记忆表，避免污染
                # 对外报告的搜索节点数与 409 终止状态
                probe_nodes = [0]

                def probe_dfs(node, budget):
                    if probe_nodes[0] > PLAN_NODE_CAP:
                        return None
                    probe_nodes[0] += 1
                    remaining = available_slots(node)
                    if not remaining:
                        reasons, _chains = validate(node)
                        return node if not reasons else None
                    for slot in remaining:
                        if not slot_candidates(node, slot):
                            return None
                    lb = lower_bound(node, remaining)
                    if lb is None or lb > budget:
                        return None

                    def _mrv_key(slot):
                        return (len(slot_candidates(node, slot)), slot[0],
                                0 if slot[1] in ("primary", "single") else 1)

                    nxt = min(remaining, key=_mrv_key)
                    room = budget - node.cable_count
                    for items, cnum, normals in slot_candidates(node, nxt):
                        if cnum > room:
                            continue
                        child, cid_map = commit_red(solver, node, items)
                        child = attach(child, nxt, items, cid_map, normals)
                        found = probe_dfs(child, budget)
                        if found is not None:
                            return found
                    return None

                if probe_dfs(Node(), hard_cap) is not None:
                    budget_limited = True
                elif probe_nodes[0] > PLAN_NODE_CAP:
                    note = (f"双通道回溯搜索超过 {PLAN_NODE_CAP} 个节点上限"
                            "（inconclusive，不提交部分方案）")
                    aborted[0] = True
        raise RedundancyConflict(
            "不存在满足全部关键路由单点断路保护的双通道联合规划：已穷尽槽位"
            "排序与替代通路，仍有关键路由的备用通路被迫与主路共用断点"
            "（或存在无法路由的请求）",
            _conflict_payload(solver, devices, pairs, routes, rinfo, slots,
                              critical, max_new, baseline_cables,
                              baseline_conns, greedy_node, best, stats,
                              budget_limited, note))

    node, chains = feasible
    return _success_payload(solver, devices, pairs, routes, rinfo, critical,
                            max_new, baseline_cables, baseline_conns,
                            node, chains, stats, note)


# ---- 成功结果序列化 -------------------------------------------------------


def _normal_standby_info(items, live_nids):
    """通路中被最终占用断开（转热备）的常态。"""
    out = []
    for kind, payload in items:
        if kind == "edge" and payload[0] == "normal":
            nid = payload[1]["id"]
            if nid not in live_nids:
                out.append(nid)
    return sorted(out)


def _success_payload(solver, devices, pairs, routes, rinfo, critical, max_new,
                     baseline_cables, baseline_conns, node, chains, stats, note):
    bind(solver, node)
    plugs = node.plugs
    live = active_normals_under(solver, pairs, plugs)

    # 按「路由序号；关键路由主先备后」的稳定顺序重编跳线号
    order_seq = []
    for idx, src, dst, crit in rinfo:
        roles = ("primary", "backup") if crit else ("single",)
        for role in roles:
            order_seq.append((idx, role))
    remap, new_cables, used_conns = {}, [], set()
    next_id = 1
    for idx, role in order_seq:
        items = node.commits[(idx, role)][0]
        for key in path_cables(items):
            if key in used_conns:
                continue
            used_conns.add(key)
            old = node.plugs[key[0]]
            remap[old] = next_id
            next_id += 1
    old_cable = {c["id"]: c for c in node.cables}
    for old_id in sorted(remap):
        c = old_cable[old_id]
        new_cables.append({
            "id": remap[old_id],
            "fromJack": c["fromJack"], "toJack": c["toJack"],
            "fromPort": c["fromPort"], "toPort": c["toPort"],
            "reusedBaseline": (c["fromJack"], c["toJack"]) in baseline_conns,
        })

    def remapped_map(items):
        m = {}
        for fj, tj in path_cables(items):
            m[(fj, tj)] = remap[node.plugs[fj]]
        return m

    # 跳线 -> 归属槽位（全局插孔唯一，一根跳线只可能属于一个槽位）
    cable_owner = {}
    routes_out = []
    for idx, src, dst, crit in rinfo:
        entry = {"index": idx, "source": src, "target": dst,
                 "critical": crit}
        if crit:
            p_items = node.commits[(idx, "primary")][0]
            b_items = node.commits[(idx, "backup")][0]
            p_map, b_map = remapped_map(p_items), remapped_map(b_items)
            p_alive = chain_alive(p_items, cable_map(new_cables), live)
            b_alive = chain_alive(b_items, cable_map(new_cables), live)
            for cid in p_map.values():
                cable_owner[cid] = {"route": idx, "role": "primary"}
            for cid in b_map.values():
                cable_owner[cid] = {"route": idx, "role": "backup"}
            entry.update({
                "primary": {
                    "chain": solver.build_chain(p_items, p_map),
                    "cables": sorted(set(p_map.values())),
                    "normalIds": sorted(path_normals(p_items)),
                    "alive": p_alive,
                    "standbyNormals": _normal_standby_info(p_items, live),
                },
                "backup": {
                    "chain": solver.build_chain(b_items, b_map),
                    "cables": sorted(set(b_map.values())),
                    "normalIds": sorted(path_normals(b_items)),
                    "alive": b_alive,
                    "standbyNormals": _normal_standby_info(b_items, live),
                },
                "cableCount": len(set(p_map.values()) | set(b_map.values())),
            })
        else:
            s_items = node.commits[(idx, "single")][0]
            s_map = remapped_map(s_items)
            for cid in s_map.values():
                cable_owner[cid] = {"route": idx, "role": "single"}
            entry.update({
                "path": {
                    "chain": solver.build_chain(s_items, s_map),
                    "cables": sorted(set(s_map.values())),
                    "normalIds": sorted(path_normals(s_items)),
                    "alive": chain_alive(s_items, cable_map(new_cables), live),
                    "standbyNormals": _normal_standby_info(s_items, live),
                },
                "cableCount": len(set(s_map.values())),
            })
        routes_out.append(entry)

    for c in new_cables:
        owner = cable_owner.get(c["id"], {})
        c["routeIndex"] = owner.get("route")
        c["role"] = owner.get("role")

    total = len(new_cables)
    extra = max(0, total - baseline_cables)

    loops = detect_loops(devices, pairs, new_cables,
                         lambda p: p["id"] in live)
    notes = []
    for c in new_cables:
        for jack in (c["fromJack"], c["toJack"]):
            n = plug_note(solver.pair_of_jack[jack], jack, solver.jack_port)
            if n and n not in notes:
                notes.append(n)
    steps = [
        f"第 {c['id']} 步：取跳线 #{c['id']}，一端插入前面板 {c['fromJack']}"
        f"（{c['fromPort']}），另一端插入 {c['toJack']}（{c['toPort']}）"
        + ("（沿用基线规划的同一跳线）" if c["reusedBaseline"] else "（新增跳线）")
        for c in new_cables
    ]

    # 逐断点幸存预览
    out_chains = {}
    for idx, src, dst, crit in rinfo:
        if crit:
            out_chains[idx] = {"critical": True, "slots": {
                "primary": node.commits[(idx, "primary")][0],
                "backup": node.commits[(idx, "backup")][0]}}
        else:
            out_chains[idx] = {"critical": False, "slots": {
                "single": node.commits[(idx, "single")][0]}}
    breakpoints, preview = simulate_breaks(solver, pairs, new_cables, out_chains)

    # 占用摘要：插孔 -> 占用跳线；常态 -> 最终是否生效及使用槽位
    jack_occ = []
    for j in sorted(plugs, key=lambda x: (int(x[:-1]), x[-1])):
        old = plugs[j]
        if old in remap:
            jack_occ.append({"jack": j, "port": solver.jack_port.get(j),
                             "cableId": remap[old]})
    normal_occ = []
    slot_normals = defaultdict(list)
    for idx, role in order_seq:
        for nid in path_normals(node.commits[(idx, role)][0]):
            slot_normals[nid].append({"route": idx, "role": role})
    for nid in sorted(slot_normals):
        pair = solver.pair_of_jack[f"{nid}T"]
        normal_occ.append({
            "pairId": nid, "mode": pair["mode"], "modeLabel": MODE_LABELS[pair["mode"]],
            "active": nid in live, "users": slot_normals[nid],
            "topPort": _pid(pair["top"]) if pair["top"] else None,
            "bottomPort": _pid(pair["bottom"]) if pair["bottom"] else None,
        })

    critical_breaks_ok = all(not f["criticalLost"] for f in preview)
    return {
        "ok": True,
        "redundant": True,
        "cableCount": total,
        "baselineCableCount": baseline_cables,
        "extraCableCount": extra,
        "maxNewCables": max_new,
        "withinBudget": extra <= max_new,
        "cables": new_cables,
        "steps": steps,
        "notes": notes,
        "routes": routes_out,
        "conflicts": [],
        "feedbackLoops": loops,
        "pairs": pairs_summary(pairs, solver),
        "criticalRoutes": critical,
        "occupancy": {"jacks": jack_occ, "normals": normal_occ},
        "failurePreview": {"breakpoints": breakpoints, "breaks": preview,
                           "criticalRoutes": critical,
                           "allCriticalSurvive": critical_breaks_ok},
        "redundancy": {"searchNodes": stats["nodes"],
                       "truncated": stats["truncated"],
                       "solverNote": note,
                       "baselineCableCount": baseline_cables},
    }


# ---- 无解负载 -------------------------------------------------------------


def _ideal_backup_blockers(solver, p_items):
    """备用理想路径（忽略一切占用）相对主路被迫共用的断点。

    返回 (blockers, witness, truncated)：
      blockers: 断点集合最小的那条理想备用路径与主路冲突的断点列表；
                若存在完全无冲突的理想备用路径则返回 None（冲突仅由联合
                插孔占用造成，不是结构性无解）；
      witness:  (items, cid_map)——理想跳线未安装，cid_map 用 9001 起的
                见证编号，供 build_chain 渲染。
    判定：① 与主路同一条路径；② 经过主路也经过的常态；③ 跳线插入主路已
    插跳线占用的插孔（独占插孔）。主路经常态、备路在该对端点插跳线构成的
    热备 / 并送**不算**冲突（拔线后常态恢复、切常态跳线仍在，由逐断点
    仿真终判）。
    """
    bind(solver, Node())
    # 主路端点：由主路端口序列反推
    ports = [payload for kind, payload in p_items if kind == "port"]
    src, dst = ports[0], ports[-1]
    paths, trunc = enumerate_paths(solver, src, dst, MAX_CABLE_DEPTH)
    p_normals = path_normals(p_items)
    p_plugs = set()
    for fj, tj in path_cables(p_items):
        p_plugs.add(fj)
        p_plugs.add(tj)

    best_spec = None
    for items, _cnum in paths:
        same = path_sig(items) == path_sig(p_items)
        blockers = []
        if same and not path_cables(items):
            blockers = [{"type": "same-path",
                         "label": "备用通路与主通路完全相同，不构成第二条通路"}]
        else:
            for nid in sorted(path_normals(items) & p_normals):
                pair = solver.pair_of_jack[f"{nid}T"]
                blockers.append({
                    "type": "normal", "pairId": nid,
                    "label": f"第 {nid} 对常态（{MODE_LABELS[pair['mode']]}，"
                             f"{_pid(pair['top'])} → {_pid(pair['bottom'])}）"
                             "被主备两条通路共用，切断即同归于尽"})
            for fj, tj in path_cables(items):
                for j in (fj, tj):
                    if same or j in p_plugs:
                        blockers.append({
                            "type": "jack", "jack": j,
                            "port": solver.jack_port.get(j),
                            "label": f"插孔 {j}（{solver.jack_port.get(j)}）"
                                     "已被主路跳线独占，备用跳线无法同时插入"})
            # 热备豁免：备路在主路常态所在对的端点插跳线（且备路本身不经过
            # 该常态）——这是合法的常态/跳线热备，不算冲突断点。
            blockers = [b for b in blockers
                        if not (b["type"] == "jack"
                                and j_normal_standby_ok(b, p_items, items))]
        if not blockers:
            return None, None, trunc
        spec = (len(blockers), path_sig(items), blockers, items)
        if best_spec is None or spec[:2] < best_spec[:2]:
            best_spec = spec
    if best_spec is None:
        return None, None, trunc
    witness_map = {key: 9001 + k
                   for k, key in enumerate(path_cables(best_spec[3]))}
    return best_spec[2], (best_spec[3], witness_map), trunc


def j_normal_standby_ok(blocker, p_items, b_items):
    """blocker 是备路在插孔 j 插线与主路争用：若 j 所在对的常态正是主路
    经过、备路不经过的常态，则该“争用”是合法热备/并送端点汇合。"""
    j = blocker.get("jack")
    if not j:
        return False
    pair_id = int(j[:-1])
    p_normals = path_normals(p_items)
    b_normals = path_normals(b_items)
    return pair_id in p_normals and pair_id not in b_normals


def _route_ideal_diagnose(solver, idx, src, dst, node):
    """终止状态下某条路由的理想路径冲突原因（对照忽略占用的理想路径）。"""
    bind(solver, node)
    found = solver.find_path(src, dst)
    if found and not solver.path_consistent(found[1]):
        found = None
    ideal = solver.find_path(src, dst, ignore_plugs=True)
    reasons, ports = diagnose(solver, src, dst, ideal, set())
    if found is None and not reasons:
        reasons = ["当前联合占用状态下不存在可与其他通路共存的通路"]
    return reasons, ports, ideal


def _conflict_payload(solver, devices, pairs, routes, rinfo, slots, critical,
                      max_new, baseline_cables, baseline_conns, greedy_node,
                      best, stats, budget_limited, note):
    node = best["node"] or Node()
    bind(solver, node)
    done = set(node.order)

    conflicts, unsatisfied = [], []
    occupied_jacks = [{"jack": j, "port": solver.jack_port.get(j),
                       "cableId": cid}
                      for j, cid in sorted(node.plugs.items(),
                                           key=lambda kv: (int(kv[0][:-1]),
                                                           kv[0][-1]))]

    for idx, src, dst, crit in rinfo:
        if crit:
            has_p = (idx, "primary") in done
            has_b = (idx, "backup") in done
            if not has_p:
                reasons, ports, ideal = _route_ideal_diagnose(
                    solver, idx, src, dst, node)
                unsatisfied.append({"index": idx, "source": src, "target": dst,
                                    "critical": True, "role": "primary",
                                    "status": "unroutable", "reasons": reasons,
                                    "ports": ports})
                continue
            if not has_b:
                p_items = node.commits[(idx, "primary")][0]
                blockers, witness, trunc = _ideal_backup_blockers(
                    solver, p_items)
                # 终止占用下的现实阻塞原因（备用通路所需插孔可能已被其他
                # 路由占尽）
                bind(solver, node)
                cur = solver.find_path(src, dst)
                if cur and not solver.path_consistent(cur[1]):
                    cur = None
                cur_reasons, cur_ports = [], []
                if cur is None:
                    ideal = solver.find_path(src, dst, ignore_plugs=True)
                    cur_reasons, cur_ports = diagnose(solver, src, dst, ideal,
                                                      set())
                witness_chain = None
                if witness is not None:
                    w_items, w_map = witness
                    witness_chain = solver.build_chain(w_items, w_map)
                conflict = {
                    "route": idx, "source": src, "target": dst,
                    "status": "no-redundant-path",
                    "blockers": blockers or [
                        {"type": "occupied",
                         "label": "存在不与主路共用断点的理想备用通路，但其"
                                  "所需插孔在联合占用下无法同时取得（独占"
                                  "插孔冲突），提高跳线预算也无法消除"}],
                    "witness": witness_chain,
                    "occupiedPorts": cur_ports,
                    "occupiedReasons": cur_reasons,
                    "primaryChain": solver.build_chain(
                        p_items, node.commits[(idx, "primary")][1]),
                    "truncated": bool(trunc),
                }
                conflicts.append(conflict)
                unsatisfied.append({
                    "index": idx, "source": src, "target": dst,
                    "critical": True, "role": "backup",
                    "status": "redundancy-conflict",
                    "reasons": [b["label"] for b in conflict["blockers"]]
                               + cur_reasons})
        else:
            if (idx, "single") not in done:
                reasons, ports, _ideal = _route_ideal_diagnose(
                    solver, idx, src, dst, node)
                unsatisfied.append({"index": idx, "source": src, "target": dst,
                                    "critical": False, "role": "single",
                                    "status": "unroutable", "reasons": reasons,
                                    "ports": ports})
    unsatisfied.sort(key=lambda u: u["index"])

    terminal_cables = [dict(c) for c in node.cables]
    payload = {
        "ok": False,
        "redundant": False,
        "inconclusive": stats["nodes"] > PLAN_NODE_CAP
                         or bool(stats["deepSlots"]),
        "solverNote": note,
        "budgetLimited": budget_limited,
        "maxNewCables": max_new,
        "baselineCableCount": baseline_cables,
        "searchNodes": stats["nodes"],
        "criticalRoutes": critical,
        "conflicts": conflicts,
        "unsatisfiedRoutes": unsatisfied,
        "occupiedJacks": occupied_jacks,
        "cables": terminal_cables,
        "pairs": pairs_summary(pairs, solver),
        "terminal": {"committedSlots": [(idx, role) for idx, role in node.order],
                     "cableCount": node.cable_count},
        "baseline": {
            "pairs": _baseline_pairs(solver, pairs, greedy_node),
            "cables": [dict(c) for c in greedy_node.cables],
            "cableCount": len(greedy_node.cables),
        },
    }
    bind(solver, node)
    return payload


def _baseline_pairs(solver, pairs, node):
    """临时切到贪心节点生成 pairs 摘要，随后切回调用方绑定状态。"""
    cur = (dict(solver.plugs), solver.cables, set(solver.protected_normals),
           solver.next_cid)
    bind(solver, node)
    out = pairs_summary(pairs, solver)
    solver.plugs, solver.cables, solver.protected_normals, solver.next_cid = cur
    return out


def _baseline_conflict_payload(solver, pairs, routes, greedy_node,
                               greedy_failed, critical):
    bind(solver, greedy_node)
    unsatisfied = []
    done = set(greedy_node.order)
    for f in greedy_failed:
        unsatisfied.append({"index": f["index"], "source": f["source"],
                            "target": f["target"], "critical": f["index"] in set(critical),
                            "role": None, "status": "baseline-unroutable",
                            "reasons": f["reasons"], "ports": []})
    for idx, r in enumerate(routes):
        if idx in done or any(u["index"] == idx for u in unsatisfied):
            continue
        src, dst = _pid(r["source"]), _pid(r["target"])
        unsatisfied.append({"index": idx, "source": src, "target": dst,
                            "critical": idx in set(critical), "role": None,
                            "status": "baseline-unroutable",
                            "reasons": ["基线规划未能为该路由提交通路"],
                            "ports": []})
    unsatisfied.sort(key=lambda u: u["index"])
    return {
        "ok": False,
        "redundant": False,
        "baselineInvalid": True,
        "criticalRoutes": critical,
        "conflicts": [],
        "unsatisfiedRoutes": unsatisfied,
        "occupiedJacks": [{"jack": j, "port": solver.jack_port.get(j),
                           "cableId": cid}
                          for j, cid in sorted(greedy_node.plugs.items(),
                                               key=lambda kv: (int(kv[0][:-1]),
                                                               kv[0][-1]))],
        "cables": [dict(c) for c in greedy_node.cables],
        "pairs": pairs_summary(pairs, solver),
        "baseline": {"pairs": pairs_summary(pairs, solver),
                     "cables": [dict(c) for c in greedy_node.cables],
                     "cableCount": len(greedy_node.cables)},
    }
