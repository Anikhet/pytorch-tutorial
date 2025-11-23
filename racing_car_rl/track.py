import pygame
import math

class Track:
    """Race track environment with boundaries and checkpoints."""

    def __init__(self, width=1200, height=800):
        self.width = width
        self.height = height

        # Define track boundaries (outer and inner walls)
        self.outer_boundary = self._create_outer_boundary()
        self.inner_boundary = self._create_inner_boundary()

        # Starting position and angle for cars
        self.start_pos = (600, 650)
        self.start_angle = -90  # Facing up

        # Colors
        self.grass_color = (34, 139, 34)
        self.track_color = (80, 80, 80)
        self.line_color = (255, 255, 255)
        self.boundary_color = (255, 0, 0)

    def _create_outer_boundary(self):
        """Create outer boundary of the track (oval shape)."""
        # Define points for outer boundary (clockwise)
        center_x, center_y = self.width // 2, self.height // 2
        width_radius = 500
        height_radius = 300

        points = []
        num_points = 100
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            x = center_x + width_radius * math.cos(angle)
            y = center_y + height_radius * math.sin(angle)
            points.append((x, y))

        return points

    def _create_inner_boundary(self):
        """Create inner boundary of the track."""
        center_x, center_y = self.width // 2, self.height // 2
        width_radius = 300
        height_radius = 150

        points = []
        num_points = 100
        for i in range(num_points):
            angle = (i / num_points) * 2 * math.pi
            x = center_x + width_radius * math.cos(angle)
            y = center_y + height_radius * math.sin(angle)
            points.append((x, y))

        return points

    def render(self, screen):
        """Render the track on the screen."""
        # Draw grass background
        screen.fill(self.grass_color)

        # Draw track surface (area between boundaries)
        # First draw a filled polygon for the outer boundary
        pygame.draw.polygon(screen, self.track_color, self.outer_boundary)
        # Then draw the inner boundary as grass to create the hole
        pygame.draw.polygon(screen, self.grass_color, self.inner_boundary)

        # Draw boundary lines
        pygame.draw.polygon(screen, self.boundary_color, self.outer_boundary, 3)
        pygame.draw.polygon(screen, self.boundary_color, self.inner_boundary, 3)

        # Draw center dashed line (optional, for aesthetics)
        self._draw_center_line(screen)

        # Draw start line
        pygame.draw.line(screen, self.line_color, (550, 650), (650, 650), 5)

    def _draw_center_line(self, screen):
        """Draw dashed center line on track."""
        center_x, center_y = self.width // 2, self.height // 2
        width_radius = 400
        height_radius = 225

        num_dashes = 50
        for i in range(0, num_dashes, 2):  # Every other segment
            angle1 = (i / num_dashes) * 2 * math.pi
            angle2 = ((i + 1) / num_dashes) * 2 * math.pi

            x1 = center_x + width_radius * math.cos(angle1)
            y1 = center_y + height_radius * math.sin(angle1)
            x2 = center_x + width_radius * math.cos(angle2)
            y2 = center_y + height_radius * math.sin(angle2)

            pygame.draw.line(screen, self.line_color, (x1, y1), (x2, y2), 2)

    def is_on_track(self, x, y):
        """Check if a point (x, y) is on the track (between boundaries)."""
        return self._point_inside_polygon(x, y, self.outer_boundary) and \
               not self._point_inside_polygon(x, y, self.inner_boundary)

    def _point_inside_polygon(self, x, y, polygon):
        """Check if point is inside polygon using ray casting algorithm."""
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def get_boundaries(self):
        """Return both boundaries for collision detection."""
        return self.outer_boundary, self.inner_boundary
