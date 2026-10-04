"""关键路由双通道规划（critical-route redundant / dual-path planning）。

用户勾选**必须抗单点断路**的关键路由，并提交**最多新增跳线数**（相对输入顺序
贪心规划的跳线数）。系统沿现有设备、插孔对与全 / 半 / 直通常态语义，为**全部
路由**联合选择可**同时安装**的主通路与备用通路：

  - 每条关键路由得到两条通路（primary / backup），二者**不得共用会被拔断的
    跳线（同一根前面板跳线、或两根争用同一独占插孔的跳线）、被切断的常态
    （同一插孔对的常态触点）**；
  - **半常态上插孔并送按最终占用判定**：半常态上插孔插线不断常态，备用跳线
    可以从上插孔并送取信号——切常态时上插孔仍直接连背板源端，主 / 备互不
    影响；但下插孔插线会切断常态，被切的常态只能由**同一根**对方通路跳线
    在拔除后恢复（单断点可恢复的“休眠常态”）；
  - 非关键路由仍需在最终状态下保持可用，其生效常态受保护，后续跳线不得
    掐断；
  - 优化目标字典序：**全部路由先满足 → 最终安装跳线总数最少**（即新增跳线
    最少）→ 输入顺序优先、候选路径按跳线数升序。

搜索模型（确定性 DFS + 跳线预算迭代加深，与 ``isolation.py`` 同构）：

  1. 状态为不可变节点 ``DNode(plugs, cables, protected_normals, commits)``；
  2. 路由按输入序号固定顺序提交（终态可行性与提交顺序无关，固定顺序不丢解）；
  3. 每条路由在**当前节点状态**下用环无关 DFS 枚举“当前可通声”的候选路径
     （复用 ``isolation.enumerate_paths``，含插孔占用与常态保护限制）；
  4. 关键路由在候选路径两两组合上做**联合占用校验**：合并主备跳线插头后按
     最终插头占用重算每条常态是否生效，把常态分成 **live（终态生效）** 与
     **dormant（被对方通路同一根跳线压住、拔除即恢复）**；
  5. 全部提交后在终态上做**逐一断点仿真**（每根已安装跳线拔除 / 每条生效
     常态强制切断），保证每个断点之后每条关键路由至少一条通路仍可实现。

无解（或搜索超限不伪称无解）时抛 :class:`RedundancyConflict`（409），
携带：**冲突断点**（输入顺序贪心方案中关键路由单链路上的每一根跳线 / 每一条
常态都是未被备份的单点）、**占用插孔**、**未满足路由及原因**、贪心对照拓扑
与搜索终止状态——**不返回任何接线步骤，不提交部分方案**。
"""

from collections import defaultdict

from solver import (FULL, HALF, THROUGH, MODE_LABELS, ConfigError, Solver,
                    detect_loops, pairs_summary, parse_config,
                    plug_note, plan_all, _pid)
from isolation import enumerate_paths

PLAN_NODE_CAP = 120_000        # 回溯搜索节点上限（超出返回 inconclusive）
PATH_ENUM_CAP = 40             # 每条路由在每个状态下保留的候选路径上限
MAX_CABLE_DEPTH = 4            # 单条通路候选路径最多经过的跳线数


class RedundancyError(ConfigError):
    """双通道请求非法（400 REDUNDANCY_INVALID）。"""

    def __init__(self, message, details=None):
        super().__init__(message, code="REDUNDANCY_INVALID",
                         details=details or {})


class RedundancyConflict(ConfigError):
    """不存在满足全部主备约束的完整规划（409 REDUNDANCY_CONFLICT）。"""

    def __init__(self, message, payload):
        super().__init__(message, code="REDUNDANCY_CONFLICT", details=payload)
        self.payload = payload


# ---- 请求解析 -------------------------------------------------------------


def parse_request(raw, n_routes):
    """校验关键路由声明与新增跳线预算，返回 (critical_set, max_add)。"""
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise RedundancyError(
            'redundancy 必须是 JSON 对象，形如 {"criticalRoutes": [0, 2], '
            '"maxNewCables": 4}')

    def _as_int(x, ctx):
        if isinstance(x, bool) or not isinstance(x, (int, float)):
            raise RedundancyError(f"{ctx} 必须是非负整数")
        if not float(x).is_integer():
            raise RedundancyError(f"{ctx} 必须是非负整数")
        return int(x)

    crit_raw = raw.get("criticalRoutes", [])
    if not isinstance(crit_raw, list):
        raise RedundancyError("criticalRoutes 必须是路由序号数组，如 [0, 2]")
    critical = set()
    for i, x in enumerate(crit_raw):
        if isinstance(x, bool):
            raise RedundancyError(f"criticalRoutes[{i}] 必须是路由序号整数")
        idx = _as_int(x, f"criticalRoutes[{i}]")
        if idx < 0 or idx >= n_routes:
            raise RedundancyError(
                f"criticalRoutes[{i}] = {idx} 越界：当前共 {n_routes} 条路由"
                f"（序号 0..{n_routes - 1}）",
                details={"index": idx, "routeCount": n_routes})
        if idx in critical:
            raise RedundancyError(f"路由 {idx + 1} 被重复勾选为关键路由")
        critical.add(idx)

    max_add = raw.get("maxNewCables", None)
    if max_add is None:
        max_add = None                            # 不限制
    else:
        max_add = _as_int(max_add, "maxNewCables")
        if max_add < 0:
            raise RedundancyError("maxNewCables 必须是非负整数")
    return critical, max_add


# ---- 搜索状态 -------------------------------------------------------------


class DNode:
    """一次联合规划前缀的完整物理状态（不可变，分支时整体复制）。"""

    __slots__ = ("plugs", "cables", "protected", "next_cid", "commits", "order")

    def __init__(self, plugs=None, cables=None, protected=None,
                 next_cid=1, commits=None, order=None):
        self.plugs = plugs or {}                # jack -> 跳线编号
        self.cables = cables or []              # [{id, fromJack, toJack, ...}]
        self.protected = protected or set()     # 已提交路由终态生效的常态编号
        self.next_cid = next_cid
        self.commits = commits if commits is not None else {}
        self.order = order or []

    @property
    def cable_count(self):
        return len(self.cables)


def bind(solver, node):
    """把搜索状态挂到 Solver 上以复用常态判定 / 邻居枚举 / 链路构造。"""
    solver.plugs = node.plugs
    solver.cables = node.cables
    solver.protected_normals = node.protected
    solver.next_cid = node.next_cid


def _path_resources(items):
    """提取一条候选路径的常态编号、跳线边（有序）与全部跳线插孔。"""
    normals, cables, jacks = [], [], set()
    for kind, payload in items:
        if kind != "edge":
            continue
        if payload[0] == "normal":
            normals.append(payload[1]["id"])
        elif payload[0] == "cable":
            fj, tj = payload[1], payload[2]
            cables.append((fj, tj))
            jacks.add(fj)
            jacks.add(tj)
    return normals, cables, jacks


def _normal_active(pair_id, pair_of, plugs, forced=()):
    """按最终插头占用判定常态是否生效（half：仅下插孔插线断）。"""
    if pair_id in forced:
        return False
    p = pair_of[f"{pair_id}T"]
    if p["mode"] == THROUGH or not (p["top"] and p["bottom"]):
        return False
    t, b = f"{pair_id}T", f"{pair_id}B"
    if p["mode"] == FULL:
        return t not in plugs and b not in plugs
    return b not in plugs                     # HALF：上插孔插线并送，不断


def _breaker_plugs(pair_id, pair_of, plugs):
    """在 plugs 占用下导致该常态不生效的插头插孔列表。"""
    p = pair_of[f"{pair_id}T"]
    t, b = f"{pair_id}T", f"{pair_id}B"
    out = []
    if p["mode"] == FULL:
        if t in plugs:
            out.append(t)
        if b in plugs:
            out.append(b)
    elif p["mode"] == HALF:
        if b in plugs:
            out.append(b)
    return out


# 组合校验失败原因（中文说明，也用于无解诊断）
REASON_LABELS = {
    "shared-normal": "主备两条通路共用同一条常态连接（切断该常态则同时失效）",
    "shared-cable": "主备两条通路共用同一根前面板跳线（拔除则同时失效）",
    "occupied-jack": "主备跳线争用同一独占插孔（一个插孔物理上只能插一根跳线）",
    "dormant-breaker-other": "备用常态已被其他路由的跳线切断，单断点无法恢复",
    "dormant-own-plug": "通路自身跳线切断了自身依赖的常态（拔掉主/备跳线都无法恢复）",
    "dormant-multi": "休眠常态被多根跳线压住，拔一根跳线无法完整恢复该通路",
    "breaks-protected": "主备跳线会切断此前路由已占用并受保护的生效常态",
    "baseline-silent": "主备两条通路在终态都不通声（至少一条必须当下可用）",
}


def validate_pair(solver, node, items_a, items_b):
    """校验同一关键路由的两条候选路径能否联合安装。

    返回 (info, reasons)：成功时 info 为提交所需的完整描述（含每条通路的
    live / dormant 常态分类、跳线映射与终态可通声标记）；失败时 reasons 为
    原因代码列表。
    """
    reasons = []
    ns_a, cables_a, jacks_a = _path_resources(items_a)
    ns_b, cables_b, jacks_b = _path_resources(items_b)

    # ---- 独占插孔 / 共用跳线 -------------------------------------------
    if set(cables_a) & set(cables_b):
        reasons.append("shared-cable")
    shared_jacks = jacks_a & jacks_b
    if shared_jacks:
        reasons.append("occupied-jack")

    # ---- 同一条常态触点不能同时被主备依赖（切常态会同时杀死两条） ------
    shared_normals = set(ns_a) & set(ns_b)
    if shared_normals:
        reasons.append("shared-normal")

    # ---- 现有连接 / 插孔占用归类 ----------------------------------------
    existing_edge = { (c["fromJack"], c["toJack"]): c["id"] for c in node.cables }
    new_a, new_b = [], []
    jack_owner_edge = {}                    # 新跳线插孔 -> (fj,tj) 归属

    def classify(cables, own_new, branch):
        for e in cables:
            if e in existing_edge:
                continue
            fj, tj = e
            if fj in node.plugs or tj in node.plugs:
                # 枚举理论上已排除（被已有跳线占用），这里兜底
                reasons.append("occupied-jack")
                continue
            own_new.append(e)
            for j in (fj, tj):
                if j in jack_owner_edge and jack_owner_edge[j] != e:
                    reasons.append("occupied-jack")
                jack_owner_edge[j] = e

    classify(cables_a, new_a, "A")
    classify(cables_b, new_b, "B")

    # ---- 合并插头占用，按最终占用重算常态 ------------------------------
    union_plugs = dict(node.plugs)
    for e in new_a + new_b:
        fj, tj = e
        union_plugs.setdefault(fj, None)
        union_plugs.setdefault(tj, None)
    union_plugs = set(union_plugs)

    # 新跳线不得切断此前路由（含非关键 / 其他关键路由）受保护的生效常态
    for j in union_plugs - set(node.plugs):
        pair = solver.pair_of_jack[j]
        if pair["id"] in node.protected:
            if pair["mode"] == FULL or j.endswith("B"):
                reasons.append("breaks-protected")

    def classify_normals(ns, own_cables, mate_cables):
        """把本通路常态分为 live / dormant，返回 (live, dormant, override)。"""
        own_edges = set(own_cables)
        mate_edges = set(mate_cables)
        live, dormant, override = set(), set(), None
        for nid in ns:
            if _normal_active(nid, solver.pair_of_jack, union_plugs):
                live.add(nid)
                continue
            dormant.add(nid)
            breakers = _breaker_plugs(nid, solver.pair_of_jack, union_plugs)
            breaker_edges = set()
            bad = False
            for j in breakers:
                if j in node.plugs:
                    reasons.append("dormant-breaker-other")
                    bad = True
                    continue
                e = jack_owner_edge.get(j)
                if e is None:
                    reasons.append("dormant-breaker-other")
                    bad = True
                    continue
                if e in own_edges:
                    reasons.append("dormant-own-plug")
                    bad = True
                    continue
                if e not in mate_edges:
                    reasons.append("dormant-breaker-other")
                    bad = True
                    continue
                breaker_edges.add(e)
            if not bad:
                if len(breaker_edges) != 1:
                    reasons.append("dormant-multi")
                else:
                    e = next(iter(breaker_edges))
                    if override is not None and override != e:
                        # 同一通路的多个休眠常态必须由同一根对方跳线压住，
                        # 否则拔掉任一根对方跳线都不能完整恢复本通路
                        reasons.append("dormant-multi")
                    override = e
        return live, dormant, override

    live_a, dormant_a, override_a = classify_normals(ns_a, new_a, new_b)
    live_b, dormant_b, override_b = classify_normals(ns_b, new_b, new_a)

    audible_a = not dormant_a
    audible_b = not dormant_b
    if not audible_a and not audible_b:
        reasons.append("baseline-silent")

    if reasons:
        return None, sorted(set(reasons))

    # ---- 分配跳线编号（A 通路先、B 通路边，确定性） --------------------
    cid_map, cables_out, added = {}, [], 0
    for e in list(dict.fromkeys(new_a + new_b)):
        cid = node.next_cid + len(cables_out)
        fj, tj = e
        cables_out.append({
            "id": cid, "fromJack": fj, "toJack": tj,
            "fromPort": solver.jack_port[fj], "toPort": solver.jack_port[tj],
        })
        cid_map[e] = cid
    for e, cid in existing_edge.items():
        cid_map[e] = cid

    def branch_info(items, live, dormant, override_edge, audible):
        return {
            "items": items,
            "cidMap": dict(cid_map),
            "liveNormals": set(live),
            "dormantNormals": set(dormant),
            "overrideEdge": override_edge,
            "overrideCableId": cid_map[override_edge] if override_edge else None,
            "audibleFinal": audible,
        }

    info = {
        "newCables": cables_out,
        "cableMap": cid_map,
        "liveNormals": live_a | live_b,
        "branches": {
            "A": branch_info(items_a, live_a, dormant_a, override_a, audible_a),
            "B": branch_info(items_b, live_b, dormant_b, override_b, audible_b),
        },
    }
    return info, []


def commit_pair(node, info):
    """按校验结果把一对主备通路提交到新节点（返回新 DNode）。"""
    plugs = dict(node.plugs)
    cables = [dict(c) for c in node.cables]
    for c in info["newCables"]:
        cables.append(dict(c))
        plugs[c["fromJack"]] = c["id"]
        plugs[c["toJack"]] = c["id"]
    protected = set(node.protected) | set(info["liveNormals"])
    new = DNode(plugs, cables, protected,
                node.next_cid + len(info["newCables"]),
                dict(node.commits), list(node.order))
    return new


def commit_single(solver, node, items):
    """提交一条非关键路由路径（0-1 最短路候选，语义同 Solver.commit）。"""
    plugs = dict(node.plugs)
    cables = [dict(c) for c in node.cables]
    protected = set(node.protected)
    cid_map, added = {}, 0
    for kind, payload in items:
        if kind != "edge":
            continue
        if payload[0] == "normal":
            protected.add(payload[1]["id"])
        elif payload[0] == "cable":
            fj, tj = payload[1], payload[2]
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
    new = DNode(plugs, cables, protected, node.next_cid + added,
                dict(node.commits), list(node.order))
    normals, _c, _j = _path_resources(items)
    return new, cid_map, set(normals)


# ---- 终态仿真 -------------------------------------------------------------


def realize(items, present_edges, pair_of, plugs, forced=()):
    """断链仿真：该通路在给定插头 / 强制切断常态下是否完整可实现。"""
    for kind, payload in items:
        if kind != "edge":
            continue
        if payload[0] == "device":
            continue
        if payload[0] == "cable":
            if (payload[1], payload[2]) not in present_edges:
                return False
        elif payload[0] == "normal":
            if not _normal_active(payload[1]["id"], pair_of, plugs, forced):
                return False
    return True


# ---- 主搜索 ---------------------------------------------------------------


def plan_redundancy(config, req_raw):
    """为全部路由联合规划主备通路。成功返回结果 dict；失败抛 RedundancyConflict。"""
    devices, pairs, routes, port_jack = parse_config(config)
    critical, max_add = parse_request(req_raw, len(routes))
    solver = Solver(devices, pairs, routes, port_jack)
    pair_of = solver.pair_of_jack

    # ---- 输入顺序贪心对照（原规划）：基线跳线数 + 无解时冲突分析 --------
    base_result, base_plan = plan_all(config)
    baseline_cable_count = len(base_plan["solver"].cables)
    total_cap = baseline_cable_count + max_add if max_add is not None else len(port_jack)

    rinfo = [(i, _pid(r["source"]), _pid(r["target"])) for i, r in enumerate(routes)]
    all_idx = set(range(len(routes)))
    stats = {"nodes": 0, "truncated": False}
    best = {"node": None, "remaining": None, "complete": False, "reasons": None}
    memo_paths = {}

    def state_key(node):
        return (frozenset(node.plugs.items()), frozenset(node.protected))

    def candidates(node, idx):
        key = (state_key(node), idx)
        if key not in memo_paths:
            bind(solver, node)
            _i, src, dst = rinfo[idx]
            paths, trunc = enumerate_paths(solver, src, dst, MAX_CABLE_DEPTH,
                                           cap=PATH_ENUM_CAP)
            if trunc:
                stats["truncated"] = True
            memo_paths[key] = paths
        return memo_paths[key]

    def record(node, remaining, complete, reasons=None):
        if best["node"] is None or len(remaining) < len(best["remaining"] or []):
            best.update(node=node, remaining=sorted(remaining),
                        complete=complete, reasons=reasons)

    def pair_options(node, idx):
        """关键路由的 (items_p, items_q, info) 候选，按新增跳线数升序。"""
        paths = candidates(node, idx)
        out = []
        seen = set()
        for a in range(len(paths)):
            for b in range(len(paths)):
                if a == b:
                    continue
                ia, _ca = paths[a]
                ib, _cb = paths[b]
                info, reasons = validate_pair(solver, node, ia, ib)
                if info is None:
                    continue
                ba, bb = info["branches"]["A"], info["branches"]["B"]
                # 主通路规范化：终态可通声者优先；并列时跳线少、键序小者优先
                def role_key(br, items):
                    return (0 if br["audibleFinal"] else 1,
                            sum(1 for k, p in items if k == "edge" and p[0] == "cable"))
                if role_key(bb, ib) < role_key(ba, ia):
                    primary = (ib, bb)
                    backup = (ia, ba)
                else:
                    primary = (ia, ba)
                    backup = (ib, bb)
                pkey = (tuple(_edge_sig(primary[0])), tuple(_edge_sig(backup[0])))
                rpkey = tuple(sorted(pkey))
                if rpkey in seen:
                    continue
                seen.add(rpkey)
                added = len(info["newCables"])
                out.append((added, pkey, primary, backup, info))
        out.sort(key=lambda x: (x[0], x[1]))
        return out

    abort = [False]

    def dfs(node, budget):
        if abort[0]:
            return None
        stats["nodes"] += 1
        if stats["nodes"] > PLAN_NODE_CAP:
            abort[0] = True
            return None

        remaining = sorted(all_idx - set(node.order))
        if not remaining:
            bind(solver, node)
            if verify_leaf(solver, node, pair_of, critical):
                return node
            record(node, remaining, True)
            return None

        # 必要条件剪枝：占用与常态保护单调增长，当前不可通 / 不足两条则永不可行
        for idx in remaining:
            paths = candidates(node, idx)
            if not paths:
                record(node, remaining, False)
                return None
            if idx in critical and len(paths) < 2 and not stats["truncated"]:
                record(node, remaining, False)
                return None

        for nxt in remaining:
            room = budget - node.cable_count
            if nxt in critical:
                opts = pair_options(node, nxt)
                if not opts:
                    continue
                progressed = False
                for added, _key, primary, backup, info in opts:
                    if added > room:
                        break
                    child = commit_pair(node, info)
                    entry = {
                        "kind": "pair",
                        "primary": primary[1], "backup": backup[1],
                        "primaryItems": primary[0], "backupItems": backup[0],
                    }
                    child.commits[nxt] = entry
                    child.order.append(nxt)
                    progressed = True
                    found = dfs(child, budget)
                    if found is not None or abort[0]:
                        return found
                if not progressed:
                    record(node, remaining, False, pair_reject_reasons(solver, node, nxt))
            else:
                paths = candidates(node, nxt)
                depth = min(MAX_CABLE_DEPTH, room)
                found_here = False
                for items, cnum in paths:
                    if cnum > depth:
                        break
                    child, cid_map, normals = commit_single(solver, node, items)
                    child.commits[nxt] = {
                        "kind": "single", "items": items,
                        "cidMap": cid_map, "liveNormals": normals,
                    }
                    child.order.append(nxt)
                    found_here = True
                    found = dfs(child, budget)
                    if found is not None or abort[0]:
                        return found
                if not found_here:
                    record(node, remaining, False)
        return None

    feasible = None
    for budget in range(0, total_cap + 1):
        feasible = dfs(DNode(), budget)
        if feasible is not None or abort[0]:
            break

    note = None
    inconclusive = abort[0]
    if abort[0]:
        note = (f"联合回溯搜索超过 {PLAN_NODE_CAP} 个节点上限：已列出搜索中发现的"
                "主备冲突，但未能在时限内证明不存在可行规划（inconclusive），未提交任何方案")
    elif stats["truncated"]:
        inconclusive = True
        note = ("部分状态的候选路径枚举触及上限：可能存在未枚举到的主备替代通路，"
                "结果标记为 inconclusive，未提交任何方案")

    if feasible is None:
        raise RedundancyConflict(
            "不存在满足全部关键路由单点断路保护的双通道规划：已按最终插头占用"
            "穷尽主备通路组合与跳线预算，仍有关键路由找不到资源互不重叠、"
            "可同时安装的两条通路（或存在无法路由的请求）",
            _conflict_payload(solver, devices, pairs, routes, critical,
                              base_result, base_plan, best, stats, total_cap,
                              baseline_cable_count, inconclusive, note))

    return _success_payload(solver, devices, pairs, routes, critical, feasible,
                            baseline_cable_count, max_add, total_cap, stats, note)


def _edge_sig(items):
    """路径的可哈希边签名（用于候选去重）。"""
    sig = []
    for kind, payload in items:
        if kind != "edge":
            continue
        if payload[0] == "normal":
            sig.append(("n", payload[1]["id"]))
        elif payload[0] == "cable":
            sig.append(("c", payload[1], payload[2]))
        else:
            sig.append(("d", payload[1]))
    return sig


def pair_reject_reasons(solver, node, idx):
    """无解诊断：统计该关键路由候选路径两两组合的失败原因分布。"""
    # 直接复用主搜索的枚举结果不太现实（节点不同），这里做一次轻量枚举
    from solver import _pid as _p
    src = None
    # rinfo 不在此函数作用域，改由调用处信息兜底：从 solver.routes 取
    r = solver.routes[idx]
    src, dst = _p(r["source"]), _p(r["target"])
    paths, _t = enumerate_paths(solver, src, dst, MAX_CABLE_DEPTH,
                                cap=min(PATH_ENUM_CAP, 24))
    counts = defaultdict(int)
    for a in range(len(paths)):
        for b in range(a + 1, len(paths)):
            _info, reasons = validate_pair(solver, node, paths[a][0], paths[b][0])
            for code in reasons:
                counts[code] += 1
    return dict(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


def verify_leaf(solver, node, pair_of, critical):
    """终态全量校验：所有路由终态可用，且逐一断点后关键路由至少一通路幸存。"""
    present_edges = {(c["fromJack"], c["toJack"]) for c in node.cables}
    plugs = set(node.plugs)

    # 基线：每条已提交路由至少其声明路径可实现
    for idx, entry in node.commits.items():
        if entry["kind"] == "single":
            if not realize(entry["items"], present_edges, pair_of, plugs):
                return False
        else:
            ok_p = realize(entry["primaryItems"], present_edges, pair_of, plugs)
            ok_b = realize(entry["backupItems"], present_edges, pair_of, plugs)
            if not ok_p and not ok_b:
                return False
            if entry["primary"]["audibleFinal"] != ok_p:
                return False
            if entry["backup"]["audibleFinal"] != ok_b:
                return False

    # 逐一断点：拔除任一跳线 / 切断任一终态生效常态
    scenarios = [("none", None)]
    for c in node.cables:
        scenarios.append(("cable", c))
    for p in solver.pairs:
        if _normal_active(p["id"], pair_of, plugs):
            scenarios.append(("normal", p))

    for stype, obj in scenarios:
        if stype == "none":
            f_edges, f_plugs, forced = present_edges, plugs, ()
        elif stype == "cable":
            e = (obj["fromJack"], obj["toJack"])
            f_edges = present_edges - {e, (e[1], e[0])}
            f_plugs = plugs - {obj["fromJack"], obj["toJack"]}
            forced = ()
        else:
            f_edges, f_plugs, forced = present_edges, plugs, (obj["id"],)
        for idx in critical:
            entry = node.commits.get(idx)
            if entry is None:
                return False
            ok_p = realize(entry["primaryItems"], f_edges, pair_of, f_plugs, forced)
            ok_b = realize(entry["backupItems"], f_edges, pair_of, f_plugs, forced)
            if not ok_p and not ok_b:
                return False
    return True


# ---- 结果序列化 -----------------------------------------------------------


def _branch_out(solver, br, items, role):
    chain = solver.build_chain(items, br["cidMap"])
    cable_ids = sorted(set(_cable_ids(items, br["cidMap"])))
    return {
        "role": role,
        "chain": chain,
        "cables": cable_ids,
        "normalIds": sorted(br["liveNormals"] | br["dormantNormals"]),
        "liveNormalIds": sorted(br["liveNormals"]),
        "dormantNormalIds": sorted(br["dormantNormals"]),
        "overrideCableId": br["overrideCableId"],
        "audibleFinal": br["audibleFinal"],
    }


def _cable_ids(items, cid_map):
    for kind, payload in items:
        if kind == "edge" and payload[0] == "cable":
            e = (payload[1], payload[2])
            if e in cid_map:
                yield cid_map[e]


def _failure_preview(solver, node, critical):
    """枚举全部断点，返回每个断点后关键路由的幸存通路与受影响路由。"""
    pair_of = solver.pair_of_jack
    present_edges = {(c["fromJack"], c["toJack"]) for c in node.cables}
    plugs = set(node.plugs)

    def audible(entries, edges, p2, forced):
        out = {}
        for idx, entry in entries:
            if entry["kind"] == "single":
                out[idx] = realize(entry["items"], edges, pair_of, p2, forced)
            else:
                out[idx] = (
                    realize(entry["primaryItems"], edges, pair_of, p2, forced)
                    or realize(entry["backupItems"], edges, pair_of, p2, forced))
        return out

    entries = sorted(node.commits.items())
    base_aud = audible(entries, present_edges, plugs, ())

    points = []
    scenarios = [c for c in node.cables] + \
                [p for p in solver.pairs if _normal_active(p["id"], pair_of, plugs)]
    for obj in scenarios:
        stype = "cable" if isinstance(obj, dict) and "fromJack" in obj else "normal"
        if stype == "cable":
            e = (obj["fromJack"], obj["toJack"])
            edges = present_edges - {e, (e[1], e[0])}
            fplugs = plugs - {obj["fromJack"], obj["toJack"]}
            forced = ()
            label = (f"跳线 #{obj['id']}：{obj['fromJack']}（{obj['fromPort']}）"
                     f" ↔ {obj['toJack']}（{obj['toPort']}）")
            tid, ports = obj["id"], [
                {"jack": obj["fromJack"], "port": obj["fromPort"]},
                {"jack": obj["toJack"], "port": obj["toPort"]}]
        else:
            edges, fplugs, forced = present_edges, plugs, (obj["id"],)
            label = (f"常态连接 第{obj['id']}对（{MODE_LABELS[obj['mode']]}）："
                     f"{_pid(obj['top'])} → {_pid(obj['bottom'])}")
            tid, ports = obj["id"], [
                {"jack": f"{obj['id']}T", "port": _pid(obj["top"])},
                {"jack": f"{obj['id']}B", "port": _pid(obj["bottom"])}]
        after = audible(entries, edges, fplugs, forced)
        affected = sorted(i for i, ok in after.items() if base_aud.get(i) and not ok)

        crit_out = []
        for idx in sorted(critical):
            entry = node.commits[idx]
            roles = []
            survivor_role, survivor_items = None, None
            if realize(entry["primaryItems"], edges, pair_of, fplugs, forced):
                roles.append("primary")
                survivor_role, survivor_items = "primary", entry["primaryItems"]
            if realize(entry["backupItems"], edges, pair_of, fplugs, forced):
                roles.append("backup")
                if survivor_role is None:
                    survivor_role, survivor_items = "backup", entry["backupItems"]
            chain = solver.build_chain(
                survivor_items, entry[survivor_role]["cidMap"]) if survivor_role else None
            switched = (survivor_role == "backup"
                        and realize(entry["primaryItems"], present_edges, pair_of, plugs))
            crit_out.append({
                "route": idx, "survivingRoles": roles,
                "survivorRole": survivor_role, "survivorChain": chain,
                "switchedToBackup": switched,
            })

        points.append({
            "type": stype, "id": tid, "label": label, "ports": ports,
            "affectedCount": len(affected), "affectedRoutes": affected,
            "critical": crit_out,
        })

    points.sort(key=lambda p: (-p["affectedCount"],
                               0 if p["type"] == "cable" else 1, p["id"]))
    return points


def _occupancy(solver, node):
    """汇总跳线 / 插孔 / 常态的最终占用（含主备角色与半常态并送标记）。"""
    cable_users = defaultdict(list)           # cid -> [{route, role}]
    normal_users = defaultdict(list)          # pair id -> [{route, role, state}]

    for idx in sorted(node.commits):
        entry = node.commits[idx]
        if entry["kind"] == "single":
            for cid in _cable_ids(entry["items"], entry["cidMap"]):
                cable_users[cid].append({"route": idx, "role": "single"})
            for nid in entry["liveNormals"]:
                normal_users[nid].append({"route": idx, "role": "single", "state": "live"})
        else:
            for role, br, items in (
                    ("primary", entry["primary"], entry["primaryItems"]),
                    ("backup", entry["backup"], entry["backupItems"])):
                for cid in _cable_ids(items, br["cidMap"]):
                    cable_users[cid].append({"route": idx, "role": role})
                state_of = lambda n: "live" if n in br["liveNormals"] else "dormant"
                for nid in br["liveNormals"] | br["dormantNormals"]:
                    normal_users[nid].append(
                        {"route": idx, "role": role, "state": state_of(nid),
                         "overrideCableId": br["overrideCableId"]})

    cables_out = []
    for c in node.cables:
        roles = {u["role"] for u in cable_users.get(c["id"], [])}
        if roles == {"primary"}:
            role_label, kind = "主通路", "primary"
        elif roles == {"backup"}:
            role_label, kind = "备用", "backup"
        elif roles == {"single"}:
            role_label, kind = "非关键", "single"
        else:
            role_label, kind = "主备共用 / 多用", "shared"
        cables_out.append({
            "id": c["id"], "fromJack": c["fromJack"], "toJack": c["toJack"],
            "fromPort": c["fromPort"], "toPort": c["toPort"],
            "users": cable_users.get(c["id"], []),
            "roleKind": kind, "roleLabel": role_label,
        })

    plugs = set(node.plugs)
    normals_out = []
    for nid in sorted(normal_users):
        p = solver.pair_of_jack[f"{nid}T"]
        active = _normal_active(nid, solver.pair_of_jack, plugs)
        override = None
        for u in normal_users[nid]:
            if u.get("overrideCableId"):
                override = u["overrideCableId"]
        mult = (p["mode"] == HALF and active and f"{nid}T" in plugs)
        normals_out.append({
            "pairId": nid, "mode": p["mode"], "modeLabel": MODE_LABELS[p["mode"]],
            "topPort": _pid(p["top"]), "bottomPort": _pid(p["bottom"]),
            "active": active, "halfMult": mult,
            "dormant": not active, "overrideCableId": override,
            "users": normal_users[nid],
        })

    return {"cables": cables_out, "normals": normals_out}


def _success_payload(solver, devices, pairs, routes, critical, node,
                     baseline_cable_count, max_add, total_cap, stats, note):
    bind(solver, node)

    routes_out = []
    for idx, r in enumerate(routes):
        src, dst = _pid(r["source"]), _pid(r["target"])
        entry = node.commits[idx]
        if entry["kind"] == "single":
            chain = solver.build_chain(entry["items"], entry["cidMap"])
            routes_out.append({
                "index": idx, "source": src, "target": dst,
                "critical": False, "status": "ok",
                "primary": {
                    "role": "single", "chain": chain,
                    "cables": sorted(set(_cable_ids(entry["items"], entry["cidMap"]))),
                    "normalIds": sorted(entry["liveNormals"]),
                    "liveNormalIds": sorted(entry["liveNormals"]),
                    "dormantNormalIds": [], "overrideCableId": None,
                    "audibleFinal": True,
                },
                "backup": None,
            })
        else:
            routes_out.append({
                "index": idx, "source": src, "target": dst,
                "critical": True, "status": "ok",
                "primary": _branch_out(solver, entry["primary"],
                                       entry["primaryItems"], "primary"),
                "backup": _branch_out(solver, entry["backup"],
                                      entry["backupItems"], "backup"),
            })

    loops = detect_loops(devices, pairs, node.cables, solver.normal_active)
    notes = []
    for c in node.cables:
        for jack in (c["fromJack"], c["toJack"]):
            n = plug_note(solver.pair_of_jack[jack], jack, solver.jack_port)
            if n and n not in notes:
                notes.append(n)
    steps = [
        f"第 {c['id']} 步：取跳线 #{c['id']}，一端插入前面板 {c['fromJack']}"
        f"（{c['fromPort']}），另一端插入 {c['toJack']}（{c['toPort']}）"
        for c in node.cables
    ]

    preview = _failure_preview(solver, node, critical)
    occupancy = _occupancy(solver, node)

    return {
        "ok": True,
        "redundant": True,
        "cableCount": node.cable_count,
        "baselineCableCount": baseline_cable_count,
        "newCableCount": node.cable_count - baseline_cable_count,
        "maxNewCables": max_add,
        "cableCap": total_cap,
        "cables": [dict(c) for c in node.cables],
        "steps": steps,
        "notes": notes,
        "routes": routes_out,
        "conflicts": [],
        "feedbackLoops": loops,
        "pairs": pairs_summary(pairs, solver),
        "occupancy": occupancy,
        "failurePreview": preview,
        "criticalRoutes": sorted(critical),
        "redundancy": {
            "searchNodes": stats["nodes"],
            "truncated": stats["truncated"],
            "solverNote": note,
            "routeOrder": list(node.order),
            "criticalCount": len(critical),
        },
    }


# ---- 无解负载 -------------------------------------------------------------


def _conflict_payload(solver, devices, pairs, routes, critical,
                      base_result, base_plan, best, stats, total_cap,
                      baseline_cable_count, inconclusive, note):
    """409 结构化负载：冲突断点 + 占用插孔 + 未满足路由（不提交部分方案）。"""
    base_s = base_plan["solver"]
    base_cables = [dict(c) for c in base_s.cables]
    base_pairs = pairs_summary(pairs, base_s)

    # 贪心方案中关键路由的单链路资源：每一根跳线 / 每一条常态都是冲突断点
    break_map = {}

    def add_point(ttype, cid, label, route_idx, ports):
        key = (ttype, cid)
        if key not in break_map:
            break_map[key] = {"type": ttype, "id": cid, "label": label,
                             "ports": ports, "criticalRoutes": []}
        if route_idx not in break_map[key]["criticalRoutes"]:
            break_map[key]["criticalRoutes"].append(route_idx)

    greedy_unroutable = sorted(c["route"] for c in base_plan["conflicts"])
    for info in base_plan["planned"]:
        if info["index"] not in critical or not info["ok"]:
            continue
        for cid in info["cableIds"]:
            c = next(x for x in base_s.cables if x["id"] == cid)
            add_point("cable", cid,
                      f"跳线 #{cid}：{c['fromJack']}（{c['fromPort']}）"
                      f" ↔ {c['toJack']}（{c['toPort']}）",
                      info["index"],
                      [{"jack": c["fromJack"], "port": c["fromPort"]},
                       {"jack": c["toJack"], "port": c["toPort"]}])
        for nid in info["normals"]:
            p = base_s.pair_of_jack[f"{nid}T"]
            add_point("normal", nid,
                      f"常态连接 第{nid}对（{MODE_LABELS[p['mode']]}）："
                      f"{_pid(p['top'])} → {_pid(p['bottom'])}",
                      info["index"],
                      [{"jack": f"{nid}T", "port": _pid(p["top"])},
                       {"jack": f"{nid}B", "port": _pid(p['bottom'])}])
    conflict_points = sorted(break_map.values(),
                             key=lambda x: (-len(x["criticalRoutes"]),
                                            0 if x["type"] == "cable" else 1,
                                            x["id"]))

    # 占用插孔（贪心方案）
    occupied_jacks = [
        {"jack": j, "cableId": cid, "port": base_s.jack_port.get(j)}
        for j, cid in sorted(base_s.plugs.items(), key=lambda kv: kv[0])
    ]

    # 未满足路由
    unsatisfied = []
    for idx, r in enumerate(routes):
        src, dst = _pid(r["source"]), _pid(r["target"])
        is_crit = idx in critical
        conflict = next((c for c in base_plan["conflicts"] if c["route"] == idx), None)
        if conflict:
            unsatisfied.append({
                "index": idx, "source": src, "target": dst,
                "critical": is_crit, "status": "unroutable",
                "reasons": conflict["reasons"], "ports": conflict["ports"],
            })
        elif is_crit:
            unsatisfied.append({
                "index": idx, "source": src, "target": dst,
                "critical": True, "status": "single-point-of-failure",
                "reasons": ["输入顺序贪心只给出一条通路：其链路上的每根跳线 / "
                            "每条常态都是未被备份的单点断点"],
                "ports": [],
            })

    # 搜索终止节点上的补充诊断（候选路径两两组合失败原因分布）
    terminal_reasons = {}
    term_node = best["node"]
    if term_node is not None:
        bind(solver, term_node)
        for idx in (best["remaining"] or []):
            if idx in critical:
                terminal_reasons[idx] = pair_reject_reasons(solver, term_node, idx)
        term_cables = [dict(c) for c in term_node.cables]
        term_plugs = [{"jack": j, "cableId": cid}
                      for j, cid in sorted(term_node.plugs.items(), key=lambda kv: kv[0])]
        term_pairs = pairs_summary(pairs, solver)
    else:
        term_cables, term_plugs, term_pairs = [], [], base_pairs

    reason_digest = []
    for idx, counts in terminal_reasons.items():
        for code, n in counts.items():
            reason_digest.append({
                "route": idx, "code": code, "count": n,
                "message": REASON_LABELS.get(code, code),
            })

    return {
        "ok": False,
        "redundant": False,
        "inconclusive": inconclusive,
        "solverNote": note,
        "searchNodes": stats["nodes"],
        "cableCap": total_cap,
        "baselineCableCount": baseline_cable_count,
        "criticalRoutes": sorted(critical),
        "conflictBreakPoints": conflict_points,
        "occupiedJacks": occupied_jacks,
        "unsatisfiedRoutes": unsatisfied,
        "reasonDigest": reason_digest,
        "greedyUnroutable": greedy_unroutable,
        "pairs": term_pairs,
        "cables": term_cables,
        "terminal": {
            "complete": best["complete"],
            "committedRoutes": sorted(term_node.order) if term_node is not None else [],
            "cableCount": term_node.cable_count if term_node is not None else 0,
            "plugs": term_plugs,
        },
        "baseline": {
            "ok": base_result["ok"],
            "cableCount": len(base_cables),
            "pairs": base_pairs,
            "cables": base_cables,
            "unroutable": [
                {"index": c["route"], "reasons": c["reasons"], "ports": c["ports"]}
                for c in base_plan["conflicts"]
            ],
        },
    }
