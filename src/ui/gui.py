import numpy as np
import pygame

from src.game.errors import GameOver
from src.game.player import Player
from src.game.game import Game, AIGame


class GUI:
    def __init__(self, game: Game, screen=None, debug: bool = None):
        if screen is None:
            self.WIDTH, self.HEIGHT = 800, 900
            self.WIN = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        else:
            self.WIN = screen
            self.WIDTH, self.HEIGHT = self.WIN.get_size()
        pygame.display.set_caption("Checkers")

        self.game = game
        self.game_over = False
        self.debug = debug

    def draw_piece(self, x, y, player: Player, king: bool):
        color = 255 if player == Player.RED else 0
        pygame.draw.circle(self.WIN, (color, 0, 0), (x * 100 + 50, y * 100 + 50), 40)
        if king:
            pygame.draw.polygon(
                self.WIN,
                (255, 255, 0),
                [
                    (x * 100 + 30, y * 100 + 30),
                    (x * 100 + 30, y * 100 + 60),
                    (x * 100 + 70, y * 100 + 60),
                    (x * 100 + 70, y * 100 + 30),
                    (x * 100 + 50, y * 100 + 50),
                ],
            )

    def draw(self):
        if not self.game_over:
            self.draw_board()
        else:
            self.WIN.fill((0, 0, 0))
            winner = self.game.winner
            font = pygame.font.Font(None, 36)
            if winner is None:
                text = font.render("Stalemate!", True, (255, 255, 255))
            else:
                text = font.render(
                    f"Game over! {self.game.winner} wins!", True, (255, 255, 255)
                )
            text_rect = text.get_rect(center=(400, 400))
            self.WIN.blit(text, text_rect)
            text = font.render(
                f"Final score: {self.game.score_str()}", True, (255, 255, 255)
            )
            text_rect = text.get_rect(center=(400, 450))
            self.WIN.blit(text, text_rect)
            pygame.display.update()

    def draw_board(self):
        # Draw the board
        x, y = pygame.mouse.get_pos()
        for i in range(8):
            for j in range(8):
                if (i + j) % 2 == 0:
                    # highlight the square the mouse is hovering over
                    if (
                        i * 100 <= x <= i * 100 + 100 and j * 100 <= y <= j * 100 + 100
                    ) or (self.game.from_square == (i, j)):
                        pygame.draw.rect(
                            self.WIN, (255, 255, 0), (i * 100, j * 100, 100, 100)
                        )
                    else:
                        pygame.draw.rect(
                            self.WIN, (255, 255, 255), (i * 100, j * 100, 100, 100)
                        )
                else:
                    pygame.draw.rect(self.WIN, (0, 0, 0), (i * 100, j * 100, 100, 100))

                # Draw the pieces
                if (piece := self.game.game_board.board[i, j]) != 0:
                    self.draw_piece(i, j, np.sign(piece), np.abs(piece) == 2)

                # Draw coordinates
                if self.debug:
                    font = pygame.font.Font(None, 24)
                    text = font.render(f"{i}, {j}", True, (0, 150, 0))
                    text_rect = text.get_rect(center=(i * 100 + 50, j * 100 + 50))
                    self.WIN.blit(text, text_rect)

        # Draw HUD at the bottom
        pygame.draw.rect(self.WIN, (0, 0, 0), (0, 800, 800, 100))
        # Print the current player
        font = pygame.font.Font(None, 36)
        text = font.render(
            f"Current player: {self.game.current_player}", True, (255, 255, 255)
        )
        text_rect = text.get_rect(center=(400, 825))
        self.WIN.blit(text, text_rect)
        # Print the number of pieces
        red, red_king = self.game.count_pieces(Player.RED)
        black, black_king = self.game.count_pieces(Player.BLACK)

        text = font.render(
            f"Red: {red} ({red_king} kings) | Black: {black} ({black_king} kings)",
            True,
            (255, 255, 255),
        )
        text_rect = text.get_rect(center=(400, 850))
        self.WIN.blit(text, text_rect)

        pygame.display.update()

    def handle_click(self, x, y):
        self.game.on_square_click(x // 100, y // 100)
        self.game_over = self.game.is_game_over()

    def tick(self):
        try:
            self.game.tick()
        except GameOver:
            self.game_over = True


if __name__ == "__main__":
    pygame.init()
    game = AIGame()
    gui = GUI(game=game, debug=False)

    run = True
    while run:
        gui.draw()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                gui.handle_click(x, y)
        if not gui.game_over:
            gui.tick()
    pygame.quit()
