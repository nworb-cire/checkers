import asyncio

import numpy as np
import pygame

pygame.init()
running = True

screen = pygame.display.set_mode((800, 900))
clock = pygame.time.Clock()


async def main():
    from src.ui.gui import GUI

    gui = GUI(screen=screen)
    global running

    while running:
        gui.draw()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                gui.handle_click(x, y)
        await asyncio.sleep(0)


asyncio.run(main())
