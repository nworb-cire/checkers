import numpy as np
import pygame

from src.game.board import Player
from src.game.game import Game


class GUI:
    def __init__(self):
        self.WIDTH, self.HEIGHT = 800, 900
        self.WIN = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Checkers")

        self.game = Game()

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
                    if (i * 100 <= x <= i * 100 + 100 and j * 100 <= y <= j * 100 + 100) or (self.game.selected_square == (i, j)):
                        pygame.draw.rect(self.WIN, (255, 255, 0), (i * 100, j * 100, 100, 100))
                    else:
                        pygame.draw.rect(self.WIN, (255, 255, 255), (i * 100, j * 100, 100, 100))
                else:
                    pygame.draw.rect(self.WIN, (0, 0, 0), (i * 100, j * 100, 100, 100))

                # Draw the pieces
                if (piece := self.game.game_board.board[i, j]) != 0:
                    self.draw_piece(i, j, np.sign(piece), np.abs(piece) == 2)

        # Draw HUD at the bottom
        pygame.draw.rect(self.WIN, (0, 0, 0), (0, 800, 800, 100))
        # Print the current player
        font = pygame.font.Font(None, 36)
        text = font.render(f"Current player: {self.game.current_player_str()}", True, (255, 255, 255))
        text_rect = text.get_rect(center=(400, 825))
        self.WIN.blit(text, text_rect)
        # Print the number of pieces
        red, red_king = self.game.count_pieces(Player.RED)
        black, black_king = self.game.count_pieces(Player.BLACK)

        text = font.render(f"Red: {red} ({red_king} kings) | Black: {black} ({black_king} kings)", True, (255, 255, 255))
        text_rect = text.get_rect(center=(400, 850))
        self.WIN.blit(text, text_rect)

        pygame.display.update()

    def handle_click(self, x, y):
        self.game.on_square_click(x // 100, y // 100)


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
                gui.handle_click(x, y)
    pygame.quit()
