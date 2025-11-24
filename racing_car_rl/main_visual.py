"""
Enhanced Racing Car Simulation with Neural Network Visualization
Shows what the car's brain is thinking in real-time!
"""

import pygame
import numpy as np
import matplotlib.pyplot as plt
from track import Track
from car import Car
from genetic_algorithm import GeneticAlgorithm
import sys

# Configuration
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 900
FPS = 60
SIMULATION_SPEED = 1  # 1x speed for better observation
MAX_GENERATION_TIME = 10  # seconds per generation
POPULATION_SIZE = 5
CAR_SPEED_MULTIPLIER = 0.4  # Slow down cars for better observation

# Colors
BACKGROUND_COLOR = (34, 139, 34)
TEXT_COLOR = (255, 255, 255)
PANEL_BG = (40, 40, 40)
SENSOR_COLOR = (255, 255, 0)

def draw_neural_network_panel(screen, car, network):
    """Draw a live visualization of the neural network."""
    panel_x = 1050
    panel_y = 50
    panel_width = 520
    panel_height = 800

    # Background panel
    pygame.draw.rect(screen, PANEL_BG, (panel_x, panel_y, panel_width, panel_height))
    pygame.draw.rect(screen, TEXT_COLOR, (panel_x, panel_y, panel_width, panel_height), 2)

    # Title
    font_title = pygame.font.Font(None, 32)
    title = font_title.render("Neural Network Live View", True, TEXT_COLOR)
    screen.blit(title, (panel_x + 80, panel_y + 10))

    font = pygame.font.Font(None, 24)
    font_small = pygame.font.Font(None, 20)

    y_offset = panel_y + 60

    # Get sensor data
    sensor_data = car.get_sensor_data()

    # SECTION 1: Sensor Inputs
    section_title = font.render("📡 SENSORS (Input Layer)", True, (100, 200, 255))
    screen.blit(section_title, (panel_x + 20, y_offset))
    y_offset += 35

    sensor_angles = ["-60°", "-30°", " 0°", "+30°", "+60°"]
    for i, (angle, value) in enumerate(zip(sensor_angles, sensor_data)):
        # Sensor label
        label = font_small.render(f"Sensor {angle}:", True, TEXT_COLOR)
        screen.blit(label, (panel_x + 30, y_offset))

        # Value bar
        bar_width = int(value * 200)
        bar_color = (int((1 - value) * 255), int(value * 255), 0)  # Red to green
        pygame.draw.rect(screen, bar_color, (panel_x + 140, y_offset + 2, bar_width, 18))
        pygame.draw.rect(screen, TEXT_COLOR, (panel_x + 140, y_offset + 2, 200, 18), 1)

        # Value text
        value_text = font_small.render(f"{value:.2f}", True, TEXT_COLOR)
        screen.blit(value_text, (panel_x + 350, y_offset))

        y_offset += 25

    y_offset += 20

    # SECTION 2: Hidden Layer 1
    section_title = font.render("🧠 HIDDEN LAYER 1", True, (100, 255, 100))
    screen.blit(section_title, (panel_x + 20, y_offset))
    y_offset += 35

    # Compute hidden layer 1
    import torch
    with torch.no_grad():
        input_tensor = torch.FloatTensor(sensor_data)
        layer1 = network.network[0](input_tensor)
        hidden1 = torch.relu(layer1).numpy()

    # Show top 4 most active neurons
    top_indices = np.argsort(hidden1)[-4:][::-1]
    for idx in top_indices:
        label = font_small.render(f"Neuron {idx+1}:", True, TEXT_COLOR)
        screen.blit(label, (panel_x + 30, y_offset))

        value = hidden1[idx]
        bar_width = int(min(value * 40, 200))
        pygame.draw.rect(screen, (100, 255, 100), (panel_x + 140, y_offset + 2, bar_width, 18))
        pygame.draw.rect(screen, TEXT_COLOR, (panel_x + 140, y_offset + 2, 200, 18), 1)

        value_text = font_small.render(f"{value:.3f}", True, TEXT_COLOR)
        screen.blit(value_text, (panel_x + 350, y_offset))

        y_offset += 25

    y_offset += 20

    # SECTION 3: Output Layer
    section_title = font.render("🎯 OUTPUT (Actions)", True, (255, 100, 100))
    screen.blit(section_title, (panel_x + 20, y_offset))
    y_offset += 35

    # Get actual output
    with torch.no_grad():
        output = network.forward(sensor_data).numpy()

    acceleration = output[0]
    steering = output[1]

    # Acceleration
    accel_label = font_small.render("Acceleration:", True, TEXT_COLOR)
    screen.blit(accel_label, (panel_x + 30, y_offset))

    # Draw bar (centered at middle for -1 to +1 range)
    center_x = panel_x + 240
    if acceleration > 0:
        bar_width = int(acceleration * 100)
        pygame.draw.rect(screen, (0, 255, 0), (center_x, y_offset + 2, bar_width, 18))
        action_text = f"🚗 GAS {acceleration:.2f}"
    else:
        bar_width = int(abs(acceleration) * 100)
        pygame.draw.rect(screen, (255, 0, 0), (center_x - bar_width, y_offset + 2, bar_width, 18))
        action_text = f"🛑 BRAKE {abs(acceleration):.2f}"

    pygame.draw.rect(screen, TEXT_COLOR, (center_x - 100, y_offset + 2, 200, 18), 1)
    pygame.draw.line(screen, TEXT_COLOR, (center_x, y_offset + 2), (center_x, y_offset + 20), 2)

    action = font_small.render(action_text, True, TEXT_COLOR)
    screen.blit(action, (panel_x + 350, y_offset))

    y_offset += 30

    # Steering
    steer_label = font_small.render("Steering:", True, TEXT_COLOR)
    screen.blit(steer_label, (panel_x + 30, y_offset))

    if steering > 0:
        bar_width = int(steering * 100)
        pygame.draw.rect(screen, (0, 100, 255), (center_x, y_offset + 2, bar_width, 18))
        action_text = f"➡️ RIGHT {steering:.2f}"
    else:
        bar_width = int(abs(steering) * 100)
        pygame.draw.rect(screen, (255, 100, 0), (center_x - bar_width, y_offset + 2, bar_width, 18))
        action_text = f"⬅️ LEFT {abs(steering):.2f}"

    pygame.draw.rect(screen, TEXT_COLOR, (center_x - 100, y_offset + 2, 200, 18), 1)
    pygame.draw.line(screen, TEXT_COLOR, (center_x, y_offset + 2), (center_x, y_offset + 20), 2)

    action = font_small.render(action_text, True, TEXT_COLOR)
    screen.blit(action, (panel_x + 350, y_offset))

    y_offset += 40

    # SECTION 4: Car Stats
    section_title = font.render("📊 CAR STATUS", True, (255, 255, 100))
    screen.blit(section_title, (panel_x + 20, y_offset))
    y_offset += 35

    stats = [
        f"Speed: {car.velocity:.1f} px/s",
        f"Distance: {car.distance_traveled:.1f} px",
        f"Alive: {'✅ Yes' if car.alive else '❌ Crashed'}",
    ]

    for stat in stats:
        text = font_small.render(stat, True, TEXT_COLOR)
        screen.blit(text, (panel_x + 30, y_offset))
        y_offset += 25

def run_generation(track, genetic_algorithm, screen, clock, generation_num):
    """Run one generation with enhanced visualization."""
    population = genetic_algorithm.get_population()

    # Create cars
    cars = []
    for i, network in enumerate(population):
        hue = (i / len(population)) * 360
        color = pygame.Color(0)
        color.hsva = (hue, 100, 100, 100)
        car = Car(track.start_pos[0], track.start_pos[1],
                 track.start_angle, track, color=color[:3])
        # Slow down cars for better observation
        car.max_velocity *= CAR_SPEED_MULTIPLIER
        car.acceleration *= CAR_SPEED_MULTIPLIER
        cars.append(car)

    # Track which car to focus on (best alive car)
    focus_car_idx = 0

    frame_count = 0
    max_frames = MAX_GENERATION_TIME * FPS

    running = True
    skip_generation = False
    while running and frame_count < max_frames and not skip_generation:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return None
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    # Skip to next generation
                    skip_generation = True
                    print("   ⏭️  Skipping to next generation...")
                elif event.key == pygame.K_1:
                    focus_car_idx = 0
                elif event.key == pygame.K_2:
                    focus_car_idx = min(1, len(cars) - 1)
                elif event.key == pygame.K_3:
                    focus_car_idx = min(2, len(cars) - 1)
                elif event.key == pygame.K_4:
                    focus_car_idx = min(3, len(cars) - 1)
                elif event.key == pygame.K_5:
                    focus_car_idx = min(4, len(cars) - 1)

        # Update physics
        for _ in range(SIMULATION_SPEED):
            alive_cars = [car for car in cars if car.alive]
            if not alive_cars:
                running = False
                break

            for car, network in zip(cars, population):
                if car.alive:
                    sensor_data = car.get_sensor_data()
                    action = network.predict(sensor_data)
                    car.update(action)

        # Find best alive car
        alive_indices = [i for i, car in enumerate(cars) if car.alive]
        if alive_indices:
            focus_car_idx = max(alive_indices, key=lambda i: cars[i].distance_traveled)

        # Render
        screen.fill(BACKGROUND_COLOR)
        track.render(screen)

        # Draw all cars
        for i, car in enumerate(cars):
            if car.alive:
                car.render(screen, show_sensors=(i == focus_car_idx))

        # Draw neural network panel for focused car
        if focus_car_idx < len(cars) and cars[focus_car_idx].alive:
            draw_neural_network_panel(screen, cars[focus_car_idx], population[focus_car_idx])

        # Draw generation info
        font = pygame.font.Font(None, 36)
        gen_text = font.render(f"Generation {generation_num}", True, TEXT_COLOR)
        screen.blit(gen_text, (20, 20))

        alive_count = sum(1 for car in cars if car.alive)
        alive_text = font.render(f"Alive: {alive_count}/{len(cars)}", True, TEXT_COLOR)
        screen.blit(alive_text, (20, 60))

        # Draw timer
        time_remaining = max(0, MAX_GENERATION_TIME - (frame_count / FPS))
        timer_text = font.render(f"Time: {time_remaining:.1f}s", True, TEXT_COLOR)
        screen.blit(timer_text, (20, 100))

        # Instructions
        font_small = pygame.font.Font(None, 20)
        instructions = [
            "Press 1-5 to focus on different cars",
            "Press SPACE to skip to next generation",
            "Yellow lines show sensors (focused car only)"
        ]
        for i, inst in enumerate(instructions):
            text = font_small.render(inst, True, (200, 200, 200))
            screen.blit(text, (20, 700 + i * 25))

        pygame.display.flip()
        clock.tick(FPS)
        frame_count += 1

    # Calculate fitness
    fitness_scores = [car.distance_traveled for car in cars]
    return fitness_scores

def main():
    """Main training loop with visualization."""
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Racing Car Evolution - Neural Network Live View")
    clock = pygame.time.Clock()

    # Create track
    track = Track(1000, SCREEN_HEIGHT)

    # Initialize genetic algorithm
    ga = GeneticAlgorithm(population_size=POPULATION_SIZE, elite_size=1, mutation_rate=0.1)

    # Training loop
    generation = 1
    fitness_history = []

    print("="*60)
    print("🏎️  RACING CAR NEURAL NETWORK EVOLUTION")
    print("="*60)
    print("\nControls:")
    print("  • Press 1-5 to focus on different cars")
    print("  • Press SPACE to skip to next generation")
    print("  • Watch the right panel to see the neural network thinking!")
    print("\n" + "="*60)

    while True:
        print(f"\n🧬 Generation {generation}")

        fitness_scores = run_generation(track, ga, screen, clock, generation)

        if fitness_scores is None:  # User quit
            break

        avg_fitness = np.mean(fitness_scores)
        max_fitness = np.max(fitness_scores)
        fitness_history.append(max_fitness)

        print(f"   Best distance: {max_fitness:.1f} px")
        print(f"   Average: {avg_fitness:.1f} px")

        # Evolve (returns best network)
        best_network = ga.evolve(fitness_scores)

        generation += 1

        # Save best network every 10 generations
        if generation % 10 == 0:
            import torch
            torch.save(best_network.state_dict(), f'best_network_gen_{generation}.pth')
            print(f"   💾 Saved checkpoint")

    pygame.quit()
    print("\n🏁 Training complete!")

if __name__ == "__main__":
    main()
