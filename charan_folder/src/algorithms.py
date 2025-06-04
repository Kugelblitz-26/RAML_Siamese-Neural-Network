import networkx as nx
import heapq
import logging
def build_dependency_graph(messages):
    G = nx.DiGraph()
    for msg in messages:
        G.add_edge(msg["sender"], msg["receiver"])
    return G

def schedule_single_node(application_data, policy):
    tasks = application_data["tasks"]
    messages = application_data.get("messages", [])

    G = build_dependency_graph(messages)
    task_map = {t["id"]: t for t in tasks}
    scheduled = {}
    schedule = []
    missed_deadlines = []

    available = [t for t in G.nodes if G.in_degree(t) == 0]
    current_time = 0

    while available:
        # Sort based on policy
        if policy == "edf":
            available.sort(key=lambda t: task_map[t]["deadline"])
        elif policy == "ldf":
            available.sort(key=lambda t: -task_map[t]["deadline"])
        elif policy == "ll":
            available.sort(key=lambda t: task_map[t]["deadline"] - task_map[t]["wcet"])

        task_id = available.pop(0)
        task = task_map[task_id]

        preds = list(G.predecessors(task_id))
        if any(p in missed_deadlines for p in preds):
            missed_deadlines.append(task_id)
            G.remove_node(task_id)
            continue

        ready_time = max(scheduled[p]["end_time"] for p in preds) if preds else 0
        start_time = max(current_time, ready_time)
        end_time = start_time + task["wcet"]

        if end_time > task["deadline"]:
            missed_deadlines.append(task_id)
            G.remove_node(task_id)
            continue

        current_time = end_time
        entry = {
            "task_id": task_id,
            "node_id": 0,
            "start_time": start_time,
            "end_time": end_time,
            "deadline": task["deadline"]
        }

        scheduled[task_id] = entry
        schedule.append(entry)
        G.remove_node(task_id)

        # Add newly available tasks
        for succ in list(G.nodes):
            if (
                succ not in scheduled
                and succ not in missed_deadlines
                and all(p in scheduled for p in G.predecessors(succ))
                and succ not in available
            ):
                available.append(succ)

    return {
        "schedule": schedule,
        "missed_deadlines": missed_deadlines,
        
    }




def schedule_multi_node(application_data, platform_data, policy="edf"):
    logging.info(f"🚀 Starting {policy.upper()} Multi-node scheduling WITHOUT communication delays")

    tasks = application_data["tasks"]
    messages = application_data.get("messages", [])
    nodes = [node["id"] for node in platform_data["nodes"] if node["type"] == "compute"]

    # Build task dependency graph
    G = nx.DiGraph()
    for task in tasks:
        G.add_node(task["id"], task=task)
    for msg in messages:
        G.add_edge(msg["sender"], msg["receiver"])

    # Track number of unscheduled dependencies
    in_degrees = {task["id"]: 0 for task in tasks}
    for u, v in G.edges:
        in_degrees[v] += 1

    # Priority queue of ready tasks
    def task_priority(task_id):
        task = G.nodes[task_id]["task"]
        if policy == "edf":
            return task["deadline"]
        elif policy == "ldf":
            return -task["deadline"]
        elif policy == "ll":
            return task["deadline"] - task["wcet"]
        else:
            raise ValueError(f"Unknown policy: {policy}")

    ready_heap = []
    for task in tasks:
        if in_degrees[task["id"]] == 0:
            heapq.heappush(ready_heap, (task_priority(task["id"]), task["id"]))

    task_end_times = {}
    node_schedules = {node: [] for node in nodes}
    schedule = []

    while ready_heap:
        _, task_id = heapq.heappop(ready_heap)
        task = G.nodes[task_id]["task"]
        wcet = task["wcet"]

        # Earliest start time after all predecessors complete
        preds = list(G.predecessors(task_id))
        earliest_start = max([task_end_times[p] for p in preds], default=0)

        # Choose best available node (earliest available time)
        best_node = None
        best_start = float("inf")
        for node in nodes:
            scheduled = sorted(node_schedules[node])
            start = earliest_start
            for s, e in scheduled:
                if start + wcet <= s:
                    break
                start = max(start, e)
            if start < best_start:
                best_start = start
                best_node = node

        # Assign task
        start_time = best_start
        end_time = start_time + wcet
        node_schedules[best_node].append((start_time, end_time))
        node_schedules[best_node].sort()
        task_end_times[task_id] = end_time

        schedule.append({
            "task_id": task_id,
            "node_id": best_node,
            "start_time": start_time,
            "end_time": end_time,
            "deadline": task["deadline"],
            "execution_time": wcet
        })

        # Mark successors as ready if all their predecessors are done
        for succ in G.successors(task_id):
            in_degrees[succ] -= 1
            if in_degrees[succ] == 0:
                heapq.heappush(ready_heap, (task_priority(succ), succ))

    missed_deadlines = [entry["task_id"] for entry in schedule if entry["end_time"] > entry["deadline"]]
   
    return {
        "schedule": schedule,
        "missed_deadlines": missed_deadlines,
        
    }

# 🎯 Entrypoints
def edf_single_node(application_data):
    output = schedule_single_node(application_data, "edf")
    output["name"] = "EDF Single-node"
    return output

def ldf_single_node(application_data):
    output = schedule_single_node(application_data, "ldf")
    output["name"] = "LDF Single-node"
    return output

def ll_single_node(application_data):
    output = schedule_single_node(application_data, "ll")
    output["name"] = "LL Single-node"
    return output

def edf_multinode_no_delay(application_data, platform_data):
    output = schedule_multi_node(application_data, platform_data, "edf")
    output["name"] = "EDF Multinode(without delay)"
    return output

def ldf_multinode_no_delay(application_data, platform_data):
    output = schedule_multi_node(application_data, platform_data, "ldf")
    output["name"] = "LDF Multinode(without delay)"
    return output

def ll_multinode_no_delay(application_data, platform_data):
    output = schedule_multi_node(application_data, platform_data, "ll")
    output["name"] = "LL(without delay)"
    return output
