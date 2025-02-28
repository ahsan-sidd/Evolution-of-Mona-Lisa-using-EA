import random
import numpy as np
from PIL import Image, ImageDraw

# Constants
TARGET_IMAGE_PATH = "monahd1.JPG"  # Upload the image
CANVAS_SIZE = (128, 160)  # Small for faster evolution
POPULATION_SIZE = 120
MUTATION_RATE = 0.2
NUM_POLYGONS = 100  # Increase for more coverage

class DNA:
    def __init__(self, target_image, polygons=None):
        self.target_image = target_image
        self.polygons = polygons if polygons else [self.random_polygon() for _ in range(NUM_POLYGONS)]

    def sample_color_from_image(self, x=None, y=None, generation=0):
        """Sample an RGB color with adaptive transparency from the target image."""
        img_array = np.array(self.target_image)
        h, w, _ = img_array.shape  # Get image dimensions

        x = random.randint(0, w-1) if x is None else min(x, w-1)
        y = random.randint(0, h-1) if y is None else min(y, h-1)

        # Adjust transparency over generations (caps at 150)
        transparency = min(150, 50 + generation // 30)

        # Return color with transparency (RGBA)
        return tuple(img_array[y, x]) + (transparency,)


    def random_polygon(self, generation=0):
        """Generate a random polygon with adaptive size and color."""
        num_points = random.randint(6, 12) 
        size_factor = max(15, 40 - generation // 500) 
        # Pick a random center within the canvas
        x, y = random.randint(0, CANVAS_SIZE[0]), random.randint(0, CANVAS_SIZE[1])
        # Generate points around (x, y) within size_factor range
        points = [(x + random.randint(-size_factor, size_factor),
                y + random.randint(-size_factor, size_factor)) for _ in range(num_points)]
        # Get color from image instead of it being completely random
        # We discussed that we can start with good potential solutions instead of completely random ones
        color = self.sample_color_from_image(x, y, generation)  
        return points, color 


    def mutate(self, current_fitness, best_fitness, generation):
        """Adaptive mutation: Higher mutation when fitness is poor."""
        # Small constant added to avoid division by zero
        mutation_factor = 1.2 - (current_fitness / (best_fitness + 1e-6))
        adaptive_rate = max(0.01, MUTATION_RATE * mutation_factor)
        # Apply mutation based on probability
        if random.random() < adaptive_rate:
            index = random.randrange(len(self.polygons))  # Pick a random polygon index
            self.polygons[index] = self.random_polygon(generation)  # Replace with a new polygon

    def crossover(self, other):
        """Combine polygons from two parents."""
        split = random.randint(0, NUM_POLYGONS - 1)
        child_polygons = self.polygons[:split] + other.polygons[split:]
        return DNA(self.target_image, child_polygons)

    def render(self, generation):
        """Render DNA as an image with additive blending."""
        image = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
        for points, color in self.polygons:
            temp_image = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
            draw = ImageDraw.Draw(temp_image, "RGBA")
            draw.polygon(points, fill=color)
            # Blending mode ADD (This allows semi-transparent polygons to blend properly)
            image = Image.alpha_composite(image, temp_image)
        return image

# Fitness function - Using only Mean Squared Error
def fitness(image1, image2):
    """Calculate fitness using MSE."""
    img1 = np.array(image1.convert('RGB'), dtype=np.float64)
    img2 = np.array(image2.convert('RGB'), dtype=np.float64)
    mse = np.sum((img1 - img2) ** 2)  # Mean squared error
    return -mse  # Negative MSE, lower is better

# Main evolution function
def evolve(target_image):
    """Run genetic algorithm to evolve the image."""
    population = [DNA(target_image) for _ in range(POPULATION_SIZE)]
    best_dna = population[0]
    best_fitness = float('-inf')
    generation = 0
    save_thresholds = {1, 136, 217, 264, 301, 348, 372, 466, 493, 554, 655, 760, 810, 977, 1074, 2080, 3386, 5678, 7000, 10000,15000, 20000,25000,30000,35000,40000,45000,50000,55000,60000,65000,75000}

    while True:
        # Evaluate fitness
        scored_population = [(dna, fitness(dna.render(generation), target_image)) for dna in population]
        scored_population.sort(key=lambda x: x[1], reverse=True)

        # Select best DNA
        best_candidate = scored_population[0]
        if best_candidate[1] > best_fitness:
            best_fitness = best_candidate[1]
            best_dna = best_candidate[0]
            best_dna.render(generation).save("best_result.png")
            print(f"Generation {generation}: New best fitness: {best_fitness}")

        # Save image at specified thresholds
        if generation in save_thresholds:
            best_dna.render(generation).save(f"generation_{generation}.png")
            print(f"Saved image for generation {generation}")

        # Generate next population using crossover & mutation
        new_population = []
        # Elitism
        num_elites = POPULATION_SIZE // 5

        for _ in range(POPULATION_SIZE - num_elites):
            parent1, parent2 = random.choices(scored_population[:15], k=2)
            child = parent1[0].crossover(parent2[0])
            current_fitness = fitness(child.render(generation), target_image)
            child.mutate(current_fitness=current_fitness, best_fitness=best_fitness, generation=generation)
            new_population.append(child)

        new_population[:num_elites] = [dna[0] for dna in scored_population[:num_elites]]
        population = new_population
        generation += 1

# Run evolution
if __name__ == "__main__":
    target = Image.open(TARGET_IMAGE_PATH).resize(CANVAS_SIZE).convert("RGB")
    evolve(target)






