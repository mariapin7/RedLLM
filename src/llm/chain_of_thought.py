from src.llm.ask_llm import LLMWrapper

def ask_question_chain_of_thought(prompt, question, model="llama3.2"):
    llm = LLMWrapper(model=model)

    # Lista de ejemplos con razonamiento para cada pregunta
    examples = [
        "Q: How many nodes are there in the network?\n"
        "A: To answer this, I will look at the list of devices or nodes in the topology. Each node is usually represented by a unique identifier such as r1, r2, etc. By counting these unique node entries, I can determine the total number of nodes. For example, if the topology lists r1, r2, and r3, then there are 3 nodes.",

        "Q: How many IPv4 addresses are assigned to devices?\n"
        "A: I will examine the IP addresses listed for each device in the topology. IPv4 addresses have the format x.x.x.x (e.g., 192.168.1.1). I will ignore any IPv6 addresses, which use colons. Then, I will count all the IPv4 addresses assigned across all devices and sum them to get the total.",

        "Q: Which devices have the most IP addresses assigned?\n"
        "A: I will go through each device in the topology and count how many IP addresses are assigned to it. Then, I will compare the totals and identify the device or devices with the highest number of IP addresses assigned.",

        "Q: Are there any devices with special-purpose IP addresses (e.g., multicast addresses)?\n"
        "A: I will examine all the IP addresses assigned to the devices. Special-purpose IP addresses, such as multicast, usually fall within specific ranges (e.g., 224.0.0.0 to 239.255.255.255 for multicast). If I find any address that belongs to such a range, then the answer is Yes; otherwise, it is No.",

        "Q: Do any devices have multiple IP addresses assigned to them?\n"
        "A: I will go through each device in the topology and count how many IP addresses are assigned to it. If I find at least one device with more than one IP address, then the answer is Yes. If all devices have only one IP address, then the answer is No.",

        "Q: Are there any IPv6 addresses assigned?\n"
        "A: I will inspect all the IP addresses in the topology. IPv6 addresses have a different format from IPv4 — they contain colons (:) and hexadecimal segments. If I find at least one address that matches this format, then the answer is Yes. Otherwise, if all addresses follow the IPv4 format, the answer is No.",

        "Q: How many subnetworks are there in my network?\n"
        "A: I will extract all the IP addresses and identify the subnet each one belongs to, using the CIDR notation (e.g., /24, /30). Then I will collect the unique subnetworks by their prefixes. The total number of distinct subnets found gives the answer.",

        "Q: Is it possible to remove one subnetwork but keeping all the devices able to ping each other?\n"
        "A: I will analyze the network topology to see if all devices are connected through multiple paths. If there is redundancy, meaning that there are alternative routes between devices, then it may be possible to remove one subnetwork without breaking connectivity. Otherwise, if any device relies on a single path through one subnetwork, removing it would isolate that device",

        "Q: Which is the subnetwork that connects x1 to x2?\n"
        "A: First, I identify the IP addresses assigned to x1 and x2. Then, I compare the subnets to which each IP belongs. If both IPs fall within the same subnet (e.g., 192.168.1.1/24 and 192.168.1.2/24 → 192.168.1.0/24), that is the subnetwork that connects them directly. If they don’t share a subnet, then there is no direct subnetwork connecting them.",

        "Q: Is it possible for x1 to ping x2 without any hop? (negative)\n"
        "A: I first identify the IP addresses of x1 and x2. If both addresses belong to the same subnet (e.g., 192.168.1.1/24 and 192.168.1.2/24), and both devices are connected directly or through the same switch, then it is possible for x1 to ping x2 without any hop — the answer is yes. If they belong to different subnets or are connected through a router, then communication without any hop is not possible — the answer is no.",


    ]

    # Seleccionamos el ejemplo que corresponde a la pregunta actual
    example = examples[0]
    for ex in examples:
        if question.lower().strip().startswith(ex.split('\n')[0][3:].lower().strip()):
            example = ex
            break

    # Construimos el prompt completo
    full_prompt = f"Example:\n{example}\n\nNow, based on the following topology:\n{prompt}\n\nQ: {question}\nA:"
    return llm.ask(full_prompt)
