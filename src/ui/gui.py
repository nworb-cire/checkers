import numpy as np
import pygame

from src.game.board import GameBoard, Player


class GUI:
    def __init__(self):
        self.WIDTH, self.HEIGHT = 800, 900
        self.WIN = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Checkers")

        self.game_board = GameBoard()

    def draw_piece(self, x, y, player: Player, king: bool):
        color = 255 if player == Player.RED else 0
        pygame.draw.circle(self.WIN, (color, 0, 0), (x * 100 + 50, y * 100 + 50), 40)
        if king:
            pygame.draw.polygon(self.WIN, (255, 255, 0), [
                (x * 100 + 30, y * 100 + 30),
                (x * 100 + 30, y * 100 + 60),
                (x * 100 + 70, y * 100 + 60),
                (x * 100 + 70, y * 100 + 30),
                (x * 100 + 50, y * 100 + 50),
            ])

    def draw(self):
        # Draw the board
        x, y = pygame.mouse.get_pos()
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    # highlight the square the mouse is hovering over
                    if i * 100 <= x <= i * 100 + 100 and j * 100 <= y <= j * 100 + 100:
                        pygame.draw.rect(self.WIN, (255, 255, 0), (i * 100, j * 100, 100, 100))
                    else:
                        pygame.draw.rect(self.WIN, (255, 255, 255), (i * 100, j * 100, 100, 100))
                else:
                    pygame.draw.rect(self.WIN, (0, 0, 0), (i * 100, j * 100, 100, 100))

                # Draw the pieces
                if (piece := self.game_board.board[i, j]) != 0:
                    self.draw_piece(i, j, np.sign(piece), np.abs(piece) == 2)

        # Draw HUD at the bottom
        pygame.draw.rect(self.WIN, (0, 0, 0), (0, 800, 800, 100))
        # Print the current player
        font = pygame.font.Font(None, 36)
        text = font.render(f"Current player: {'red' if self.game_board.current_player == Player.RED else 'black'}", True, (255, 255, 255))
        text_rect = text.get_rect(center=(400, 850))
        self.WIN.blit(text, text_rect)

        pygame.display.update()


if __name__ == "__main__":
    pygame.init()
    gui = GUI()

    run = True
    while run:
        gui.draw()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                print(x // 100, y // 100)
    pygame.quit()
