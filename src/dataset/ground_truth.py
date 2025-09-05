import os
import json
from typing import List, Dict, Any
from tqdm import tqdm
import networkx as nx


class GroundTruthGenerator:
    def __init__(self, data_path: str = "data/topologias"):
        self.__data_path = data_path
        self.__questions = [
            "How many nodes are there in the network? Only answer with the number. Do not add any explanation.",
            "How many IPv4 addresses are assigned to devices? Only answer with the number. Do not add any explanation.",
            "Which devices have the most IP addresses assigned? Do not add any explanation.",
            "Are there any devices with special-purpose IP addresses (e.g., multicast addresses)? Do not add any explanation.",
            "Do any devices have multiple IP addresses assigned to them? Do not add any explanation.",
            "Are there any IPv6 addresses assigned? Do not add any explanation.",
            "How many subnetworks are there in my network? Do not add any explanation.",
            "Is it possible to remove one subnetwork but keeping all the devices able to ping each other? Do not add any explanation.",
            "Which is the subnetwork that connects x1 to x2? Do not add any explanation.",
            "Is it possible for x1 to ping x2 without any hop? Answer directly with \"yes\" or \"no\".",

            # "How many nodes are there in the network? Only answer with the number.",
            # "How many IPv4 addresses are assigned to devices? ",
            # "Which devices have the most IP addresses assigned? ",
            # "Are there any devices with special-purpose IP addresses (e.g., multicast addresses)? ",
            # "Do any devices have multiple IP addresses assigned to them? ",
            # "Are there any IPv6 addresses assigned? ",
            # "How many subnetworks are there in my network? Only answer with the number. ",
            # "Is it possible to remove one subnetwork but keeping all the devices able to ping each other? ",
            # "Which is the subnetwork that connects x1 to x2? ",
            # "Is it possible for x1 to ping x2 without any hop? Answer directly with \"yes\" or \"no\". ",

        ]

    def __get_files(self) -> List[str]:
        return [f for f in os.listdir(self.__data_path) if f.endswith(".json")]



    def find_common_subnet(self, node1_addrs, node2_addrs):
        """
        Encuentra una subred común entre las direcciones de node1 y node2,
        comparando solo direcciones IPv4 con prefijo (formato IP/prefijo).
        """
        subnet1 = {
            ip for ip in node1_addrs if "/" in ip and self.is_ipv4(ip.split("/")[0])
        }
        subnet2 = {
            ip for ip in node2_addrs if "/" in ip and self.is_ipv4(ip.split("/")[0])
        }

        common = set()
        for ip1 in subnet1:
            for ip2 in subnet2:
                if ip1.split("/")[1] == ip2.split("/")[1]:  # mismo prefijo
                    # asumimos misma subred si IPs tienen misma máscara y misma red
                    net1 = ip1.split("/")[0].rsplit(".", 1)[0]
                    net2 = ip2.split("/")[0].rsplit(".", 1)[0]
                    if net1 == net2:
                        common.add(ip1.split("/")[0] + "/" + ip1.split("/")[1])

        return common.pop() if common else "unknown"

    def is_ipv4(self, ip):
        return "." in ip

    def are_directly_connected(self, links, x1, x2):
        for link in links:
            endpoints = link.get("endpoints", [])
            if x1 in endpoints and x2 in endpoints:
                return "yes"
        return "no"

    def can_remove_subnet_and_stay_connected(self, nodes, links):
        if len(nodes) < 2:
            return "No"

        G_full = nx.Graph()
        for node in nodes:
            G_full.add_node(node["id"])
        for link in links:
            endpoints = link.get("endpoints", [])
            subnet = link.get("subnet", "unknown")
            if len(endpoints) == 2:
                G_full.add_edge(endpoints[0], endpoints[1], subnet=subnet)

        all_subnets = {link.get("subnet") for link in links if link.get("subnet")}

        for subnet in all_subnets:
            G_temp = G_full.copy()
            edges_to_remove = [
                (u, v) for u, v, d in G_temp.edges(data=True) if d.get("subnet") == subnet
            ]
            G_temp.remove_edges_from(edges_to_remove)

            if nx.is_connected(G_temp):
                return "Yes"

        return "No"


    def generate_ground_truth(self, output_path: str, file_name: str = None):
        #files = self.__get_files()
        files = [file_name] if file_name else self.__get_files()

        for file_name in tqdm(files, desc="Generating ground truth"):
            file_path = os.path.join(self.__data_path, file_name)
            with open(file_path, "r") as f:
                data = json.load(f)

            nodes = data.get("nodes", [])
            links = data.get("links", [])

            node_ids = [node.get("id") for node in nodes]
            all_addresses = [addr for node in nodes for addr in node.get("local_addresses", [])]
            ipv4_addrs = [addr for addr in all_addresses if ":" not in addr]
            ipv6_addrs = [addr for addr in all_addresses if ":" in addr]

            addr_count_per_node = {
                node["id"]: len(node.get("local_addresses", [])) for node in nodes
            }
            max_ips = max(addr_count_per_node.values()) if addr_count_per_node else 0
            most_ips_nodes = [k for k, v in addr_count_per_node.items() if v == max_ips]

            subnets = set(ip.split("/")[0] + "/" + ip.split("/")[1] for ip in ipv4_addrs)


            if len(node_ids) >= 2:
                x1, x2 = node_ids[0], node_ids[1]
                node1_addrs = next((n["local_addresses"] for n in nodes if n["id"] == x1), [])
                node2_addrs = next((n["local_addresses"] for n in nodes if n["id"] == x2), [])
                common_subnet = self.find_common_subnet(node1_addrs, node2_addrs)
                direct_connection = self.are_directly_connected(links, x1, x2)
            else:
                common_subnet = "unknown"
                direct_connection = "no"

            answers = [
                str(len(nodes)),  # 1
                str(len(ipv4_addrs)),  # 2
                ", ".join(most_ips_nodes),  # 3
                "Yes" if any(ip.startswith("224.") or ip.startswith("255.") for ip in ipv4_addrs) else "No",  # 5
                "Yes" if any(v > 1 for v in addr_count_per_node.values()) else "No",  # 6
                "Yes" if ipv6_addrs else "No",  # 7
                str(len(subnets)),  # 8
                self.can_remove_subnet_and_stay_connected(nodes, links),  # 9 (assume only 1 subnet is required)
                common_subnet,  # 10
                "yes" if direct_connection == "yes" else "no",  # 12
            ]

            gt_path = os.path.join(output_path, f"answers_{os.path.splitext(file_name)[0]}.jsonl")
            with open(gt_path, "w") as out:
                for q, a in zip(self.__questions, answers):
                    json.dump({"prompt": q, "answer": a}, out)
                    out.write("\n")


if __name__ == "__main__":
    generator = GroundTruthGenerator(data_path="data/topologias")
    generator.generate_ground_truth(output_path="data/ground_truth")
