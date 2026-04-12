import pygame
import sys
from track import Track
from car import Car
from neural_network import CarNeuralNetwork
import torch

# Configuration
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
FPS = 60

def run_demo(network_file='best_network_final.pth'):
    """
    Run a demo of the trained network.

    Args:
        network_file: Path to saved network weights
    """
    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Racing Car Demo - Trained Network")
    clock = pygame.time.Clock()

    # Create track
    track = Track(SCREEN_WIDTH, SCREEN_HEIGHT)

    # Load trained network
    print(f"Loading network from {network_file}...")
    try:
        network = CarNeuralNetwork()
        network.load_state_dict(torch.load(network_file))
        print("Network loaded successfully!")
    except FileNotFoundError:
        print(f"Error: {network_file} not found. Please train the network first.")
        return

    # Create car
    car = Car(track.start_pos[0], track.start_pos[1],
             track.start_angle, track, color=(0, 0, 255))

    # Font for UI
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    # Main loop
    running = True
    frame_count = 0

    while running:
        clock.tick(FPS)
        frame_count += 1

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Reset car
                    car = Car(track.start_pos[0], track.start_pos[1],
                            track.start_angle, track, color=(0, 0, 255))
                    frame_count = 0

        # Update car if alive
        if car.alive:
            # Get sensor data
            sensor_data = car.get_sensor_data()

            # Get action from neural network
            action = network.predict(sensor_data)

            # Update car
            car.update(action)
        else:
            # Auto-reset after a few seconds
            if frame_count % (FPS * 3) == 0:
                car = Car(track.start_pos[0], track.start_pos[1],
                        track.start_angle, track, color=(0, 0, 255))
                frame_count = 0

        # Render
        track.render(screen)
        car.render(screen, show_sensors=True)

        # Draw UI
        status = "ALIVE" if car.alive else "CRASHED"
        status_color = (0, 255, 0) if car.alive else (255, 0, 0)

        status_text = font.render(f"Status: {status}", True, status_color)
        screen.blit(status_text, (10, 10))

        distance_text = font.render(f"Distance: {car.distance_traveled:.1f}", True, (255, 255, 255))
        screen.blit(distance_text, (10, 50))

        fitness_text = font.render(f"Fitness: {car.fitness:.1f}", True, (255, 255, 255))
        screen.blit(fitness_text, (10, 90))

        # Instructions
        instructions = [
            "ESC: Quit",
            "R: Reset car"
        ]
        for i, instruction in enumerate(instructions):
            inst_text = small_font.render(instruction, True, (255, 255, 255))
            screen.blit(inst_text, (10, SCREEN_HEIGHT - 60 + i * 25))

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    # Allow user to specify network file
    network_file = 'best_network_final.pth'
    if len(sys.argv) > 1:
        network_file = sys.argv[1]

    run_demo(network_file)
