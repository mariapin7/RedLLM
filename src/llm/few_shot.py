from src.llm.ask_llm import LLMWrapper


def ask_question_few_shot(prompt, question, model="llama3.2"):
    llm = LLMWrapper(model=model)

    examples = {
        "How many nodes are there in the network?": [
            "Example:\nNetwork: A, B, C → 3 nodes\nQuestion: How many nodes are there in the network?\nAnswer: 3\n",
            "Example:\nNetwork has 5 devices connected: r1, r2, r3, r4, r5\nQuestion: How many nodes are there in the network?\nAnswer: 5\n",
            "Example:\nSimple network: router and switch only\nQuestion: How many nodes are there in the network?\nAnswer: 2\n"
        ],
        "How many IPv4 addresses are assigned to devices?": [
            "Example:\nDevice A has 2 IPs: 192.168.1.1/24 and 10.0.0.1/16. Device B has 3 IPs: 192.168.1.2/24, 10.0.0.2/16, and 172.16.0.1/30.\nTotal IPs = 2 (A) + 3 (B) = 5\nQuestion: How many IPv4 addresses are assigned to devices?\nAnswer: 5\n",
            "Example:\nDevice A has 1 IP: 10.0.0.1. Device B has 1 IP: 10.0.0.2. Device C has 1 IP: 10.0.0.3.\nTotal = 1 + 1 + 1 = 3\nQuestion: How many IPv4 addresses are assigned to devices?\nAnswer: 3\n",
            "Example:\nOnly Device A exists and it has 4 IPs: 192.168.1.1, 192.168.1.2, 10.0.0.1, 172.16.0.1.\nQuestion: How many IPv4 addresses are assigned to devices?\nAnswer: 4\n"
        ],
        "Which devices have the most IP addresses assigned?": [
            "Example:\nr1: 3 IPs, r2: 2 IPs, r3: 1 IP\nQuestion: Which devices have the most IP addresses assigned?\nAnswer: r1\n",
            "Example:\nAll devices (r1, r2, r3) have 2 IPs\nQuestion: Which devices have the most IP addresses assigned?\nAnswer: r1, r2, r3\n",
            "Example:\nr1: 1 IP, r2: 3 IPs, r3: 3 IPs\nQuestion: Which devices have the most IP addresses assigned?\nAnswer: r2, r3\n"
        ],
        "Are there any devices with special-purpose IP addresses (e.g., multicast addresses)?": [
            "Example:\nDevice r2 uses 224.0.0.1 → multicast address\nQuestion: Are there any devices with special-purpose IP addresses (e.g., multicast addresses)?\nAnswer: Yes\n",
            "Example:\nAll addresses used are standard private IPv4 (e.g., 192.168.0.0/16)\nQuestion: Are there any devices with special-purpose IP addresses (e.g., multicast addresses)?\nAnswer: No\n",
            "Example:\nr3 has 224.0.0.9, which is a multicast address\nQuestion: Are there any devices with special-purpose IP addresses (e.g., multicast addresses)?\nAnswer: Yes\n"
        ],
        "Do any devices have multiple IP addresses assigned to them?": [
            "Example:\nDevice A has two IPs: 192.168.1.1 and 10.0.0.1. Device B has one IP: 192.168.1.2.\nQuestion: Do any devices have multiple IP addresses assigned to them?\nAnswer: Yes\n",
            "Example:\nDevice A: 192.168.1.1. Device B: 10.0.0.2. Device C: 172.16.0.3. All devices have only one IP.\nQuestion: Do any devices have multiple IP addresses assigned to them?\nAnswer: No\n",
            "Example:\nDevice B has two IPs: 10.0.0.2 and 10.0.0.3. Devices A and C only have one each.\nQuestion: Do any devices have multiple IP addresses assigned to them?\nAnswer: Yes\n"
        ],
        "Are there any IPv6 addresses assigned?": [
            "Example:\nAll devices have IPv4 addresses like 192.168.1.1\nQuestion: Are there any IPv6 addresses assigned?\nAnswer: No\n",
            "Example:\nDevice A has fe80::1 and 2001:db8::1234\nQuestion: Are there any IPv6 addresses assigned?\nAnswer: Yes\n",
            "Example:\nNo address matches IPv6 format (e.g., xxxx::xxxx)\nQuestion: Are there any IPv6 addresses assigned?\nAnswer: No\n"
        ],
        "How many subnetworks are there in my network?": [
            "Example:\nDevices use 192.168.1.0/24 and 10.0.0.0/24 → 2 different subnets\nQuestion: How many subnetworks are there in my network?\nAnswer: 2\n",
            "Example:\nDevices are assigned to 192.168.1.0/24, 192.168.2.0/24, and 10.0.0.0/24 → 3 subnets in total\nQuestion: How many subnetworks are there in my network?\nAnswer: 3\n",
            "Example:\nAll devices share the 10.0.0.0/16 subnet\nQuestion: How many subnetworks are there in my network?\nAnswer: 1\n"
        ],
        "Is it possible to remove one subnetwork but keeping all the devices able to ping each other?": [
            "Example:\nThe network has only one subnetwork, so removing it would isolate all devices.\nQuestion: Is it possible to remove one subnetwork but keeping all the devices able to ping each other?\nAnswer: No\n",
            "Example:\nThe network has multiple redundant subnetworks connecting the same devices. Removing one does not break connectivity.\nQuestion: Is it possible to remove one subnetwork but keeping all the devices able to ping each other?\nAnswer: Yes\n",
            "Example:\nEach device is connected through a unique subnetwork. Removing any of them would cut off at least one device.\nQuestion: Is it possible to remove one subnetwork but keeping all the devices able to ping each other?\nAnswer: No\n"
        ],
        "Which is the subnetwork that connects x1 to x2?": [
            "Example:\nx1 has IP 10.0.0.1/24 and x2 has IP 10.0.0.2/24. Both belong to the same subnet 10.0.0.0/24.\nQuestion: Which is the subnetwork that connects x1 to x2?\nAnswer: 10.0.0.0/24\n",
            "Example:\nx1 has IP 192.168.1.5/24 and x2 has IP 192.168.1.10/24. Both are in the 192.168.1.0/24 subnet.\nQuestion: Which is the subnetwork that connects x1 to x2?\nAnswer: 192.168.1.0/24\n",
            "Example:\nx1 has IP 10.0.1.1/16 and x2 has IP 10.0.2.1/16. Both are in the larger subnet 10.0.0.0/16.\nQuestion: Which is the subnetwork that connects x1 to x2?\nAnswer: 10.0.0.0/16\n"
        ],
        "Is it possible for x1 to ping x2 without any hop?": [
            "Example:\nx1 and x2 are connected to the same switch without any intermediate device.\nQuestion: Is it possible for x1 to ping x2 without any hop?\nAnswer: yes\n",
            "Example:\nx1 and x2 are in the same subnet but are connected through a router.\nQuestion: Is it possible for x1 to ping x2 without any hop?\nAnswer: no\n",
            "Example:\nx1 has IP 10.0.0.1/24 and x2 has IP 10.0.0.2/24. Both are physically connected to the same switch.\nQuestion: Is it possible for x1 to ping x2 without any hop?\nAnswer: yes\n"
        ]
    }

    base_question = question.split("?", 1)[0] + "?"  # extrae el tipo base
    example_text = "\n\n".join(examples.get(base_question, []))

    full_prompt = f"{example_text}\n\nNow, based on the following topology:\n{prompt}\n\nQuestion: {question}\nAnswer:"
    return llm.ask(full_prompt)
