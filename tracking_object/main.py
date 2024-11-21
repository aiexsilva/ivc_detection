import pygame, sys, random
from ultralytics import YOLO
import tracking_function
from tracking_function import open_camera, object_tracking
import cv2
import time

pygame.init()

WIDTH, HEIGHT = 1280, 720

FONT = pygame.font.SysFont("Consolas", int(WIDTH / 20))
GAME_OVER_FONT = pygame.font.SysFont("Consolas", int(WIDTH / 10))

SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pong!")

CLOCK = pygame.time.Clock()

# Paddles
player = pygame.Rect(0, 0, 10, 100)
player.center = (WIDTH - 100, HEIGHT / 2)

opponent = pygame.Rect(0, 0, 10, 100)
opponent.center = (100, HEIGHT / 2)

player_score, opponent_score = 0, 0

# Ball
ball = pygame.Rect(0, 0, 20, 20)
ball.center = (WIDTH / 2, HEIGHT / 2)

x_speed, y_speed = 1, 1

capture = open_camera()

#game start time
start_time = time.time()

#reset the game
def reset_game():
    global player_score, opponent_score, x_speed, y_speed, ball, player, opponent
    player_score, opponent_score = 0, 0
    ball.center = (WIDTH / 2, HEIGHT / 2)
    x_speed, y_speed = random.choice([1, -1]), random.choice([1, -1])
    player.center = (WIDTH - 100, HEIGHT / 2)
    opponent.center = (100, HEIGHT / 2)

#display the game-over screen
def game_over_screen():
    SCREEN.fill("Black")
    game_over_text = GAME_OVER_FONT.render("GAME OVER", True, "white")
    play_again_text = FONT.render("Press SPACE to play again", True, "white")
    quit_text = FONT.render("Or ESC to quit", True, "white")
    SCREEN.blit(game_over_text, (WIDTH / 2 - game_over_text.get_width() / 2, HEIGHT / 3))
    SCREEN.blit(play_again_text, (WIDTH / 2 - play_again_text.get_width() / 2, HEIGHT / 2))
    SCREEN.blit(quit_text, (WIDTH / 2 - quit_text.get_width() / 2, HEIGHT / 2 + 80))
    pygame.display.update()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                #restart the game
                if event.key == pygame.K_SPACE:
                    reset_game()
                    return
                #quit the game
                elif event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()


#game loop
while True:

    #calculate elapsed time
    elapsed_time = time.time() - start_time

    #object tracking
    frame, top, top2 = object_tracking(capture)

    #cellPhone paddle
    if not (top[1] is None):
        #offset the y value -50 so it reaches the top 
        player.top = top[1] - 50

    #confines paddle to camera bounds
    if player.top < 0:
        player.top = 0
    if player.bottom > HEIGHT:
        player.bottom = HEIGHT

    #sportsBall paddle
    if not (top2[1] is None):
        opponent.top = top2[1] - 50

    if opponent.top < 0:
        opponent.top = 0
    if opponent.bottom > HEIGHT:
        opponent.bottom = HEIGHT

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    if ball.y >= HEIGHT:
        y_speed = -1
    if ball.y <= 0:
        y_speed = 1
    if ball.x <= 0:
        player_score += 1
        ball.center = (WIDTH / 2, HEIGHT / 2)
        x_speed, y_speed = random.choice([1, -1]), random.choice([1, -1])
    if ball.x >= WIDTH:
        opponent_score += 1
        ball.center = (WIDTH / 2, HEIGHT / 2)
        x_speed, y_speed = random.choice([1, -1]), random.choice([1, -1])
    if player.x - ball.width <= ball.x <= player.right and ball.y in range(player.top - ball.width,
                                                                           player.bottom + ball.width):
        x_speed = -1
    if opponent.x - ball.width <= ball.x <= opponent.right and ball.y in range(opponent.top - ball.width,
                                                                               opponent.bottom + ball.width):
        x_speed = 1

    #check for game-over conditions
    if player_score >= 15 or opponent_score >= 15 or elapsed_time >= 300:
        game_over_screen()
        #time reset
        start_time = time.time()

    player_score_text = FONT.render(str(player_score), True, "white")
    opponent_score_text = FONT.render(str(opponent_score), True, "white")

    ball.x += x_speed * 20
    ball.y += y_speed * 20

    SCREEN.fill("Black")

    pygame.draw.rect(SCREEN, "white", player)
    pygame.draw.rect(SCREEN, "white", opponent)
    pygame.draw.circle(SCREEN, "white", ball.center, 10)

    SCREEN.blit(player_score_text, (WIDTH / 2 + 50, 50))
    SCREEN.blit(opponent_score_text, (WIDTH / 2 - 50, 50))

    cv2.imshow("tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    cv2.setUseOptimized(True)
    cv2.ocl.setUseOpenCL(True)

    pygame.display.update()
    CLOCK.tick(60)