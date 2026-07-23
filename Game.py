import pygame
import numpy as np
import pickle

pygame.init()

# Variables for game states
current_image = None
prediction = ""
actual = ""
result = ""
index = None
round_active = False
answer_revealed = False
winner = None

running = True

player_score = 0
nn_score = 0

player_guess = None
nn_guess = None

round_active = False

round_start = 0
nn_reveal_time = 0
time_left = 0

WIDTH = 1000
HEIGHT = 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("My Game")

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

#Connect to MNIST dataset
mnist = np.loadtxt("mnist.csv", delimiter=",", skiprows=1)

labels = mnist[:, 0].astype(int)
images = mnist[:, 1:] / 255.0

#Create button
button = pygame.Rect(700, 550, 220, 60)
font = pygame.font.SysFont(None, 36)

while running:
    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        #Start new round
        if event.type == pygame.MOUSEBUTTONDOWN:

            if button.collidepoint(event.pos):

                winner = None
                result = ""

                index = np.random.randint(len(images))

                current_image = images[index].reshape(28,28)
                actual = labels[index]

                player_guess = None
                nn_guess = None

                round_active = True
                answer_revealed = False

                round_start = pygame.time.get_ticks()

                # AI waits between 500 and 2000 ms
                nn_reveal_time = round_start + np.random.randint(500,2000)

        # Player input
        if event.type == pygame.KEYDOWN:

            # Player can only answer if nobody has answered
            if round_active and not answer_revealed:

                # Check if key is 0-9
                if pygame.K_0 <= event.key <= pygame.K_9:

                    player_guess = event.key - pygame.K_0

                    winner = "Player"

                    answer_revealed = True

                    print("Player guessed:", player_guess)
    # AI thinking
    if round_active and not answer_revealed:

        current_time = pygame.time.get_ticks()

        if current_time >= nn_reveal_time:

            nn_guess = nn.predict(
                images[index].reshape(1,-1)
            )[0]

            winner = "AI"

            answer_revealed = True

            print("AI guessed:", nn_guess)
    
    # Decide winner
    if answer_revealed:

        if winner == "Player":

            if player_guess == actual:
                player_score += 1
                result = "Player Wins!"
            else:
                result = "Player Wrong!"

                # AI reveals its answer after the round
                nn_guess = nn.predict(
                    images[index].reshape(1,-1)
                )[0]


        elif winner == "AI":

            if nn_guess == actual:
                nn_score += 1
                result = "AI Wins!"
            else:
                result = "AI Wrong!"

                # Reveal player's chance was missed
                player_guess = "Too Slow"

        answer_revealed = False
        round_active = False

    if round_active and not answer_revealed:

        elapsed = pygame.time.get_ticks() - round_start

        time_left = 5 - elapsed / 1000

        if time_left <= 0:
            round_active = False
            result = "Time's up!"


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

    text = font.render("New Round", True, (255, 255, 255))
    screen.blit(text, (button.x + 25, button.y + 18))

    if current_image is not None:

        timer_text = font.render(
            f"Time: {max(0,time_left):.1f}",
            True,
            (255,255,255)
        )

        player_text = font.render(
            f"Player: {player_guess if player_guess is not None else 'Waiting...'}",
            True,
            (255,255,255)
        )

        ai_text = font.render(
            f"AI: {nn_guess if nn_guess is not None else 'Thinking...'}",
            True,
            (255,255,255)
        )

        score_text = font.render(
            f"Player {player_score} - {nn_score} AI",
            True,
            (255,255,0)
        )

        result_text = font.render(
            result,
            True,
            (0,255,0)
        )


        screen.blit(timer_text,(450,80))
        screen.blit(player_text,(450,130))
        screen.blit(ai_text,(450,180))
        screen.blit(score_text,(450,230))
        screen.blit(result_text,(450,280))





    pygame.display.update()

    

pygame.quit()