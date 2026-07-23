import pygame
import numpy as np
import pickle

pygame.init()

WIDTH = 1000
HEIGHT = 700

class NeuralNetwork:

    @staticmethod
    def load_model(filename):
            # Load the model parameters from a file
            with open(filename, 'rb') as f:
                return pickle.load(f)

    def relu(self, Z):
        return np.maximum(0, Z)

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        return expZ / np.sum(expZ, axis=1, keepdims=True)

    def feedforward(self, X):
        """
        Pass input through the network.
        """

        # Hidden layer
        Z1 = np.dot(X, self.W1) + self.b1
        A1 = self.relu(Z1)

         # Output layer
        Z2 = np.dot(A1, self.W2) + self.b2
        output = self.softmax(Z2)

        return output

    def predict(self, X):
        """
        Convert network output into a prediction.
        """

        output = self.feedforward(X)

        return np.argmax(output, axis=1)

nn = NeuralNetwork.load_model("NeuralNetwork.dat")

mnist = np.loadtxt("mnist.csv", delimiter=",", skiprows=1)

labels = mnist[:, 0].astype(int)
images = mnist[:, 1:] / 255.0

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("My Game")

#Create button
button = pygame.Rect(700, 550, 220, 60)
font = pygame.font.SysFont(None, 36)

current_image = None
prediction = ""
actual = ""
running = True

while running:
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            if button.collidepoint(event.pos):

                index = np.random.randint(len(images))

                current_image = images[index].reshape(28, 28)

                actual = labels[index]

                prediction = nn.predict(images[index].reshape(1, -1))[0]

    #Background color
    screen.fill((0, 0, 0))

    #image display
    if current_image is not None:

        scale = 10

        for y in range(28):
            for x in range(28):

                colour = int(current_image[y, x] * 255)

                pygame.draw.rect(
                    screen,
                    (colour, colour, colour),
                    (50 + x * scale,
                    50 + y * scale,
                    scale,
                    scale)
                )

    #Draw button
    pygame.draw.rect(screen, (70, 130, 180), button)

    text = font.render("Random Digit", True, (255, 255, 255))
    screen.blit(text, (button.x + 25, button.y + 18))

    if current_image is not None:

        pred_text = font.render(f"Prediction: {prediction}", True, (255,255,255))
        actual_text = font.render(f"Actual: {actual}", True, (255,255,255))

        screen.blit(pred_text, (450, 120))
        screen.blit(actual_text, (450, 170))




    pygame.display.update()

    

pygame.quit()
