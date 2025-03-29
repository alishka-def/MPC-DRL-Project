import os
import sys
import numpy as np
from lxml import etree
from typing import Optional, List


def generate_demand(route_id: str, flow_id: str, vtype_id: str, interp_times: np.ndarray, interp_demands: np.ndarray,
                    depart_lane: str, depart_speed: str, sim_begin: float, sim_step: float, sim_end: float,
                    tree_root: etree.Element, insertion_mode: Optional[str] = "Poisson"):
    """
    Generates <flow> elements in the route file, using linear interpolation
    between specified time-demand points if needed.

    Each generated flow element's id will be unique by appending a counter to flow_id.
    """
    assert len(interp_times) == len(interp_demands)
    assert interp_times[0] == sim_begin
    assert interp_times[-1] == sim_end

    counter = 0  # initialize a counter for unique flow IDs

    for i in range(len(interp_times) - 1):
        if interp_demands[i] == interp_demands[i + 1]:
            flowElement = etree.SubElement(tree_root, "flow")
            flowElement.set("id", f"{flow_id}_{counter}")
            counter += 1
            flowElement.set("type", vtype_id)
            flowElement.set("route", route_id)
            flowElement.set("begin", str(interp_times[i]))
            flowElement.set("end", str(interp_times[i + 1]))

            if insertion_mode == "Poisson":
                flowElement.set("period", "exp(" + str(np.round(interp_demands[i] / 3600.0, decimals=6)) + ")")
            elif insertion_mode == "Bernoulli":
                flowElement.set("probability", str(np.round(interp_demands[i] / 3600.0, decimals=6)))
            else:
                flowElement.set("vehsPerHour", str(interp_demands[i]))
            flowElement.set("departLane", depart_lane)
            flowElement.set("departSpeed", depart_speed)
            continue
        slope = (interp_demands[i + 1] - interp_demands[i]) / (interp_times[i + 1] - interp_times[i])
        times = np.arange(interp_times[i], interp_times[i + 1] + sim_step, sim_step)
        for j in range(len(times) - 1):
            demand_flow = slope * (times[j] - interp_times[i]) + interp_demands[i]
            flowElement = etree.SubElement(tree_root, "flow")
            flowElement.set("id", f"{flow_id}_{counter}")
            counter += 1
            flowElement.set("type", vtype_id)
            flowElement.set("route", route_id)
            flowElement.set("begin", str(times[j]))
            flowElement.set("end", str(times[j + 1]))
            if insertion_mode == "Poisson":
                flowElement.set("period", "exp(" + str(np.round(demand_flow / 3600.0, decimals=6)) + ")")
            elif insertion_mode == "Bernoulli":
                flowElement.set("probability", str(np.round(demand_flow / 3600.0, decimals=6)))
            else:
                flowElement.set("vehsPerHour", str(demand_flow))
            flowElement.set("departLane", depart_lane)
            flowElement.set("departSpeed", depart_speed)


def generate_route_file(route_demands_params: List[dict], sim_begin: float, sim_step: float, sim_end: float,
                        route_filepath: str) -> None:
    """
    Generates the demands for a time-varying demand profile and overwrites the existing route file.
    """
    parser = etree.XMLParser(remove_blank_text=True)
    t = etree.parse(route_filepath, parser)
    t_root = t.getroot()
    # Remove existing flow elements
    for e in t_root.findall('flow'):
        t_root.remove(e)
    for d in route_demands_params:
        generate_demand(
            d["route_id"], d["flow_id"], d["vtype_id"],
            np.array(d["interp_times"]), np.array(d["interp_demands"]),
            d["depart_lane"], d["depart_speed"], sim_begin, sim_step, sim_end,
            t_root, d["insertion_mode"]
        )
    t = etree.ElementTree(t_root)
    t.write(route_filepath, pretty_print=True, encoding='UTF-8', xml_declaration=True)


if __name__ == "__main__":
    # Testing the function

    rou_file = './network_example.rou.xml'

    route_demands = [
        {
            "route_id": "r_O2_D1",
            "flow_id": "f_0",
            "vtype_id": "car",
            "depart_lane": "free",
            "depart_speed": "max",
            "interp_times": [0, 540, 1260, 1800, 9000],
            "interp_demands": [500, 1500, 1500, 500, 500],
            "insertion_mode": "Poisson"
        },
        {
            "route_id": "r_O1_D1",
            "flow_id": "flow1",
            "vtype_id": "car",
            "depart_lane": "free",
            "depart_speed": "max",
            "interp_times": [0, 7200, 8100, 9000],
            "interp_demands": [3500, 3500, 1000, 1000],
            "insertion_mode": "Poisson"
        },
        {
            "route_id": "r_O1_D1",
            "flow_id": "flow2",
            "vtype_id": "car",
            "depart_lane": "free",
            "depart_speed": "20.0",
            "interp_times": [0, 7200, 8100, 9000],
            "interp_demands": [3500, 3500, 1000, 1000],
            "insertion_mode": "Poisson"
        }
    ]
    generate_route_file(
        route_demands_params=route_demands, sim_begin=0, sim_step=10, sim_end=9000, route_filepath=rou_file
    )
