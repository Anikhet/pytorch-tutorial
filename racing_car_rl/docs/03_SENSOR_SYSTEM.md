# Sensor System - How Cars "See"

## The Challenge

Cars need to know where the track boundaries are to avoid crashing. But neural networks can't "see" images directly (well, they can, but that's much more complex).

Instead, we give cars **distance sensors** - like having 5 laser rangefinders!

## What Are Raycasts?

A **raycast** is like shooting a laser beam in a direction until it hits something.

```
Car Position
    │
    │ ← Raycast in this direction
    │
    │
    │
    ◄─── Wall detected here!

Distance = length of ray
```

## Our 5-Sensor Setup

```
              Sensor 3 (0°)
                  ║
                  ║
        Sensor 2  ║  Sensor 4
       (-30°)  ╲  ║  ╱ (+30°)
                ╲ ║ ╱
   Sensor 1      ╲║╱    Sensor 5
   (-60°)  ════   🚗   ════  (+60°)
                  ▲
             Car facing up
```

**Sensor angles (relative to car's direction):**
1. **-60°** - Far left
2. **-30°** - Left-front
3. **0°** - Straight ahead
4. **+30°** - Right-front
5. **+60°** - Far right

## Why 5 Sensors?

**Too few (1-2):**
- Not enough information
- Can't detect side walls
- Blind spots

**Just right (5):**
- Good coverage (180° front arc)
- Enough info to make decisions
- Computationally efficient

**Too many (10+):**
- Redundant information
- Slower neural network
- More weights to evolve

## How Raycasting Works

### Step 1: Calculate Ray Direction

```python
def _cast_ray(self, angle):
    # Car is at (self.x, self.y) facing self.angle degrees

    # Sensor angle relative to car
    absolute_angle = self.angle + angle  # e.g., car at 45° + sensor at -30° = 15°

    # Convert to radians for math
    rad = math.radians(absolute_angle)

    # Direction vector
    dx = math.cos(rad)
    dy = math.sin(rad)
```

### Step 2: March Along the Ray

```python
    max_distance = 200  # Max sensor range
    step_size = 5       # Check every 5 pixels

    for distance in range(0, max_distance, step_size):
        # Calculate point along ray
        test_x = self.x + dx * distance
        test_y = self.y + dy * distance

        # Check if this point is off the track
        if not track.is_on_track(test_x, test_y):
            # Hit a wall!
            return distance / max_distance  # Normalize 0-1

    # Didn't hit anything within range
    return 1.0
```

### Step 3: Normalize the Distance

**Why normalize to 0-1?**

```python
# Raw distance
distance = 75  # pixels

# Normalized (divided by max range)
normalized = 75 / 200 = 0.375

# Neural networks work better with consistent ranges
# 0.0 = wall right next to car (danger!)
# 1.0 = no wall detected (safe)
```

## Sensor Reading Examples

### Scenario 1: Approaching a Wall

```
Before:
   Sensor 3 reading: 0.9  (wall far ahead)
   Action: Accelerate!

During approach:
   Sensor 3 reading: 0.5  (wall getting closer)
   Action: Maintain speed

Almost at wall:
   Sensor 3 reading: 0.1  (wall very close!)
   Action: BRAKE AND TURN!
```

### Scenario 2: Navigating a Right Turn

```
Sensors: [Left-far, Left, Center, Right, Right-far]
Reading: [0.3,       0.4,  0.7,    0.9,   0.95]

Interpretation:
- Left side: walls close (0.3, 0.4) ← Inside of turn
- Center: decent space (0.7)
- Right side: very clear (0.9, 0.95) ← Outside of turn

Decision: Steer RIGHT to follow the turn
```

### Scenario 3: Straight Track

```
Sensors: [Left-far, Left, Center, Right, Right-far]
Reading: [0.85,      0.9,  0.95,   0.9,   0.85]

Interpretation:
- All sensors show clear space
- Symmetric readings (left ≈ right)
- Car is centered on track

Decision: ACCELERATE and go straight
```

### Scenario 4: Heading Toward Wall

```
Sensors: [Left-far, Left, Center, Right, Right-far]
Reading: [0.8,       0.7,  0.15,   0.7,   0.8]

Interpretation:
- Center sensor VERY LOW (0.15) ← Danger!
- Side sensors OK
- About to crash head-on

Decision: HARD BRAKE + SHARP TURN (either direction)
```

## Collision Detection

How do we know if a point is "off track"?

### Point-in-Polygon Algorithm

```python
def is_on_track(self, x, y):
    # Track is the area BETWEEN outer and inner boundaries

    inside_outer = self._point_in_polygon(x, y, self.outer_boundary)
    inside_inner = self._point_in_polygon(x, y, self.inner_boundary)

    # On track if inside outer boundary BUT NOT inside inner boundary
    return inside_outer and not inside_inner
```

### Ray Casting Algorithm (for polygon detection)

```python
def _point_in_polygon(self, x, y, polygon):
    # Shoot a ray from point to infinity
    # Count how many times it crosses polygon edges
    # Odd crossings = inside, Even = outside

    crossings = 0
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]

        # Check if horizontal ray from (x, y) crosses this edge
        if ray_crosses_edge(x, y, x1, y1, x2, y2):
            crossings += 1

    return crossings % 2 == 1  # Odd = inside
```

## Sensor Visualization

When you run the simulation, you can see the sensors as colored lines:

```python
def _render_sensors(self, screen):
    for i, sensor_angle in enumerate(self.sensor_angles):
        distance = self.sensor_readings[i] * self.sensor_length

        # Calculate end point
        end_x = self.x + cos(angle) * distance
        end_y = self.y + sin(angle) * distance

        # Color based on distance
        # Green = far (safe)
        # Red = close (danger)
        danger_level = 1 - self.sensor_readings[i]
        red = int(255 * danger_level)
        green = int(255 * (1 - danger_level))
        color = (red, green, 0)

        # Draw ray
        pygame.draw.line(screen, color, (car_x, car_y), (end_x, end_y))
```

**Color meanings:**
- 🟢 **Green:** Wall far away, safe
- 🟡 **Yellow:** Wall at medium distance, caution
- 🔴 **Red:** Wall very close, danger!

## How Neural Network Uses Sensors

### Input Layer

```python
# Raw sensor data
sensors = [0.8, 0.6, 0.3, 0.9, 0.85]

# Fed directly to neural network
action = network.predict(sensors)
# action = [acceleration, steering]
```

### What the Network Learns

Over generations, hidden neurons specialize:

**Neuron 1 might learn:** "Front sensor danger detector"
```python
activation = ReLU(sensor[2] * -5.2)  # Activates strongly when front sensor < 0.2
```

**Neuron 2 might learn:** "Left turn detector"
```python
activation = ReLU(sensor[0] * 3.1 + sensor[1] * 2.8)  # Activates when left sensors are high
```

**Neuron 3 might learn:** "Speed safety check"
```python
activation = ReLU(sensor[0] * 1.2 + sensor[2] * 1.5 + sensor[4] * 1.2)  # Average clearance
```

## Limitations of Our Sensor System

### What It Can't Do

1. **No velocity sensing** - Doesn't know current speed
2. **No track position** - Doesn't know where it is on track
3. **No memory** - Can't remember previous positions
4. **Limited range** - Only 200 pixels ahead
5. **No rear sensors** - Can't see behind

### Why These Limitations Are OK

For this task, 5 forward sensors are enough because:
- We only need to avoid walls
- Track is relatively simple
- Cars always move forward
- Simplicity helps learning

### Advanced Extensions (Future Work)

**Add more sensors:**
```python
sensor_angles = [-90, -60, -30, 0, 30, 60, 90]  # 7 sensors
```

**Add velocity sensor:**
```python
inputs = [sensor1, sensor2, ..., sensor5, velocity]  # 6 inputs
```

**Add position tracking:**
```python
inputs = [...sensors, distance_from_start, lap_number]  # 7 inputs
```

**Add memory (LSTM network):**
```python
# Network remembers previous sensor readings
hidden_state = lstm(current_sensors, previous_hidden_state)
```

## Sensor Tuning

### Sensor Length (Currently: 200 pixels)

**Too short (50px):**
- Can't see far enough ahead
- React too late to turns
- More crashes

**Just right (200px):**
- Good preview of upcoming track
- Time to plan turns
- Not too computationally expensive

**Too long (500px):**
- Can see very far
- Might confuse network (too much info)
- Slower raycasting

### Step Size (Currently: 5 pixels)

**Too small (1px):**
- Very precise detection
- Slower computation
- Overkill for this task

**Just right (5px):**
- Fast and accurate enough
- Good balance

**Too large (20px):**
- Might miss thin obstacles
- Less precise
- Faster but less reliable

## Debug Tips

### Visualizing Sensors

Always render the best car's sensors:

```python
if cars[0].alive:  # Best car from previous generation
    cars[0]._render_sensors(screen)
```

**What to look for:**
- All red sensors? Car is trapped!
- Only one red? Car should turn away
- All green? Should accelerate
- Uneven left/right? Should steer to balance

### Common Issues

**Sensors not detecting wall:**
- Check sensor length (might be too short)
- Verify track.is_on_track() works correctly
- Check step size (might skip over walls)

**Cars ignore sensors:**
- Network weights might be random still (early generations)
- Fitness function might not reward survival
- Mutation rate too high (destroying good solutions)

**Sensors look wrong:**
- Check angle calculations (degrees vs radians)
- Verify car.angle is updated correctly
- Ensure coordinate system is consistent

## The Code

From `car.py`:

```python
class Car:
    def __init__(self, x, y, angle, track):
        self.sensor_angles = [-60, -30, 0, 30, 60]
        self.sensor_length = 200
        self.sensor_readings = [0] * 5

    def _update_sensors(self):
        """Update all sensor readings"""
        for i, sensor_angle in enumerate(self.sensor_angles):
            angle = math.radians(self.angle + sensor_angle)
            self.sensor_readings[i] = self._cast_ray(angle)

    def _cast_ray(self, angle):
        """Cast ray and return normalized distance"""
        step_size = 5
        for distance in range(0, self.sensor_length, step_size):
            x = self.x + math.cos(angle) * distance
            y = self.y + math.sin(angle) * distance

            if not self.track.is_on_track(x, y):
                return distance / self.sensor_length

        return 1.0  # No obstacle detected
```

## Key Takeaways

1. **Raycasts simulate distance sensors** like LIDAR or ultrasonic sensors

2. **5 sensors provide 180° field of view** - enough for this task

3. **Normalized readings (0-1)** work best with neural networks

4. **Color visualization helps debug** - see what the car "sees"

5. **Simple sensors are sufficient** - complexity isn't always better

6. **Real robots use similar systems** - this is how actual autonomous vehicles work!

## Next Steps

You now understand:
- ✅ Neural networks (the brains)
- ✅ Genetic algorithms (the evolution)
- ✅ Sensors (the eyes)

Next, let's see how it all comes together!

Continue to **04_LEARNING_PROCESS.md** → 📈
