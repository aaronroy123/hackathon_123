# Smart City Traffic Optimizer - Final Base Model
# Compares fixed-time vs dynamic traffic signal control
# Beginner-safe, hackathon-ready

SIM_TIME = 60          # total simulation time (seconds)
GREEN_TIME = 10        # fixed green duration (seconds)

# Average vehicle arrivals per second (simulated sensor data)
ARRIVAL_RATE = {
    "north": 1,
    "south": 1,
    "east": 2,
    "west": 2
}


def simulate_fixed():
    queue = {d: 0 for d in ARRIVAL_RATE}
    total_wait = 0
    directions = list(queue.keys())
    current_dir = 0

    for t in range(SIM_TIME):
        # Vehicles arrive
        for d in queue:
            queue[d] += ARRIVAL_RATE[d]

        # Rotate signal every GREEN_TIME seconds
        if t % GREEN_TIME == 0:
            green = directions[current_dir]
            current_dir = (current_dir + 1) % len(directions)

        # One vehicle passes on green
        if queue[green] > 0:
            queue[green] -= 1

        # Waiting time accumulates
        total_wait += sum(queue.values())

    return total_wait


def simulate_dynamic():
    queue = {d: 0 for d in ARRIVAL_RATE}
    total_wait = 0

    for t in range(SIM_TIME):
        # Vehicles arrive
        for d in queue:
            queue[d] += ARRIVAL_RATE[d]

        # Choose most congested direction
        green = max(queue, key=queue.get)

        # Dynamic green capacity (more congestion → more cars pass)
        cars_can_pass = max(1, queue[green] // 3)

        queue[green] = max(0, queue[green] - cars_can_pass)

        # Waiting time accumulates
        total_wait += sum(queue.values())

    return total_wait


if __name__ == "__main__":
    fixed_wait = simulate_fixed()
    dynamic_wait = simulate_dynamic()

    print("=== Simulation Results ===")
    print("Fixed signal wait time   :", fixed_wait)
    print("Dynamic signal wait time :", dynamic_wait)

    improvement = ((fixed_wait - dynamic_wait) / fixed_wait) * 100
    print(f"Wait time improvement    : {improvement:.2f}%")
