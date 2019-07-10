"""
Sprite Collect Coins

Simple program to show basic sprite usage.

Artwork from http://kenney.nl

If Python and Arcade are installed, this example can be run from the command line with:
python -m arcade.examples.sprite_collect_coins
"""

import arcade
import os
import numpy as np
import pyglet

# # ---- Tricky pyglet setting to allow running the game without calling app.run ---
event_loop = pyglet.app.event_loop
event_loop._legacy_setup()

# --- Constants ---
SPRITE_SCALING_BATTLESHIP = 0.5
SPRITE_SCALING_ALIEN = 0.1
SPRITE_SCALING_LASER = 1
SPRITE_SCALING_ALIEN_LASER = 0.5


ALIEN_ROW_COUNT = 4
ALIEN_COL_COUNT = 9
ALIEN_SPEED = 0.8
ALIEN_MOVING_RANGE = 100

SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
ALIEN_TOP_MARGIN = 60
ALIEN_LEFT_MARGIN = 60
SCREEN_TITLE = "Space Invader 2019"

MOVEMENT_SPEED = 5
BULLET_SPEED = 9
ALIEN_BULLET_SPEED = -3


class Player(arcade.Sprite):
    def __init__(self, life, *argv):
        super().__init__(*argv)
        self.life = life


class SpaceInvader_HoustonJ2013(arcade.Window):
    """ Our custom Window Class"""

    def __init__(self, visible=True, screen_width=SCREEN_WIDTH, screen_height=SCREEN_HEIGHT,
                screen_title=SCREEN_TITLE):
        """ Initializer """
        # Call the parent class initializer
        super().__init__(screen_width, screen_height, screen_title)

        # Set the working directory (where we expect to find files) to the same
        # directory this .py file is in. You can leave this out of your own
        # code, but it is needed to easily run the examples using "python -m"
        # as mentioned at the top of this program.
        file_path = os.path.dirname(os.path.abspath(__file__))
        os.chdir(file_path)

        self.screen_width = screen_width
        self.screen_height = screen_height

        self.screen_scale = self.screen_width / SCREEN_WIDTH

        ## Set window visibility inherited from pyglet window
        self.set_visible(visible)

        # --- Keep track of a frame count.
        # --- This is important for doing things every x frames
        self.frame_count = 0
        self.done = False

        # Variables that will hold sprite lists
        self.battleship_list = None
        self.alien_list = None

        # Player life reduction warning
        self.player_life_reduction_warning = False

        # Set up the player info
        self.battleship_sprite = None
        self.score = 0

        # Don't show the mouse cursor
        self.set_mouse_visible(False)


        ## Load Sounds for gun and hit
        self.gun_sound = arcade.sound.load_sound("sounds/LASRFIR2.WAV")
        self.hit_sound = arcade.sound.load_sound("sounds/hit.wav")
        print("finished setting up")

        ## actions space and actions
        self.actions_n = 6 ## actions up down left right shoot idle

        ## Save play pictures
        self.save_play_pic = False
        self.pic_folder = "./pics/"


    def setup(self):
        """ Set up the game and initialize the variables. """

        #setup background color
        arcade.set_background_color(arcade.color.BLACK)

        # Sprite lists
        self.battleship_list = arcade.SpriteList()
        self.alien_list = arcade.SpriteList()
        self.bullet_list = arcade.SpriteList()
        self.alien_bullet_list = arcade.SpriteList()

        # Score
        self.score = 0

        # Set up the player
        # Character image from kenney.nl
        self.battleship_sprite = Player(3, "pics/playerShip1_orange.png",
                                        SPRITE_SCALING_BATTLESHIP * self.screen_scale)
        self.battleship_sprite.center_x = int(1/3.0 * self.screen_width)
        self.battleship_sprite.center_y = 50 * self.screen_scale
        self.battleship_sprite.boundary_left = 0
        self.battleship_sprite.boundary_right = self.screen_width
        self.battleship_sprite.boundary_bottom = 0
        self.battleship_sprite.boundary_top = self.screen_height
        self.battleship_list.append(self.battleship_sprite)

        self._setup_alien()

        self.physics_engine = \
            arcade.PhysicsEnginePlatformer(self.battleship_sprite,
                                           self.alien_list,
                                           gravity_constant=0)

        self.episode = 1

    def draw_game_over(self):
        """
        Draw "Game over" across the screen.
        """
        output = "Game Over"
        arcade.draw_text(output, 240 * self.screen_scale, 400 * self.screen_scale, arcade.color.WHITE, int(54 * self.screen_scale))

        output = "Click to Enter Restart"
        arcade.draw_text(output, 310 * self.screen_scale, 300 * self.screen_scale, arcade.color.WHITE, int(24 * self.screen_scale))

    def draw_game(self):
        self.alien_list.draw()
        self.battleship_list.draw()
        self.bullet_list.draw()
        self.alien_bullet_list.draw()

        # Put the text on the screen.
        author = "Designed by HoustonJ2013"
        arcade.draw_text(author, int(self.screen_width - 300 * self.screen_scale),
                         (self.screen_height - 30 * self.screen_scale),
                         arcade.color.RED,
                         int(14 * self.screen_scale))

        # Game display
        output = "Level:%i Score:%i" % (self.episode, self.score)
        arcade.draw_text(output, int(self.screen_width / 2.0 - 150 * self.screen_scale),
                         int(self.screen_height - 50 * self.screen_scale), arcade.color.WHITE,
                         int(20 * self.screen_scale))
        # Play 1 life
        player_1 = self.battleship_sprite
        player_1_life = "Player 1: life %i" %(player_1.life)
        if self.player_life_reduction_warning:
            arcade.draw_text(player_1_life, int(10 * self.screen_scale),
                             int(self.screen_height - 50 * self.screen_scale), arcade.color.RED,
                             int(20 * self.screen_scale))
        else:
            arcade.draw_text(player_1_life, int(10 * self.screen_scale),
                             int(self.screen_height - 50 * self.screen_scale), arcade.color.GREEN,
                             int(20 * self.screen_scale))
        if self.save_play_pic == True:
            image = arcade.draw_commands.get_image()
            image.save(os.path.join(self.pic_folder, str(self.frame_count), ".png"), "PNG")

    def on_draw(self):
        """ Draw everything """
        arcade.start_render()
        if not self.done:
            self.draw_game()
        else:
            self.draw_game()
            self.draw_game_over()

    def on_key_press(self, key, modifiers):
        """Called whenever a key is pressed. """
        print(key, self.battleship_sprite.center_x, self.battleship_sprite.center_y)
        screen_boundary_margin = 10
        if key == arcade.key.UP and self.battleship_sprite.center_y < self.screen_height - screen_boundary_margin:
            self.battleship_sprite.change_y = MOVEMENT_SPEED
        elif key == arcade.key.UP and self.battleship_sprite.center_y >= self.screen_height - screen_boundary_margin:
            self.battleship_sprite.change_y = 0
        elif key == arcade.key.DOWN and self.battleship_sprite.center_y > screen_boundary_margin:
            self.battleship_sprite.change_y = -MOVEMENT_SPEED
        elif key == arcade.key.DOWN and self.battleship_sprite.center_y <= screen_boundary_margin:
            self.battleship_sprite.change_y = 0
        elif key == arcade.key.LEFT and self.battleship_sprite.center_x > screen_boundary_margin:
            self.battleship_sprite.change_x = -MOVEMENT_SPEED
        elif key == arcade.key.LEFT and self.battleship_sprite.center_x <= screen_boundary_margin:
            self.battleship_sprite.change_x = 0
        elif key == arcade.key.RIGHT and self.battleship_sprite.center_x < self.screen_width - screen_boundary_margin:
            self.battleship_sprite.change_x = MOVEMENT_SPEED
        elif key == arcade.key.RIGHT and self.battleship_sprite.center_x >= self.screen_width - screen_boundary_margin:
            self.battleship_sprite.change_x = 0
        elif key == arcade.key.SPACE:
            # Gunshot sound
            # arcade.sound.play_sound(self.gun_sound)
            # Create a bullet
            bullet = arcade.Sprite("pics/laserBlue01.png", SPRITE_SCALING_LASER * self.screen_scale)

            # The image points to the right, and we want it to point up. So
            # rotate it.
            bullet.angle = 90

            # Give the bullet a speed
            bullet.change_y = BULLET_SPEED

            # Position the bullet
            bullet.center_x = self.battleship_sprite.center_x
            bullet.center_y = self.battleship_sprite.center_y
            bullet.bottom = self.battleship_sprite.top

            # Add the bullet to the appropriate lists
            self.bullet_list.append(bullet)
        elif key == arcade.key.ENTER and self.done:
            self.setup()
            self.done = False

    def on_key_release(self, key, modifiers):
        """Called when the user releases a key. """

        if key == arcade.key.UP or key == arcade.key.DOWN:
            self.battleship_sprite.change_y = 0
        elif key == arcade.key.LEFT or key == arcade.key.RIGHT:
            self.battleship_sprite.change_x = 0


    def update(self, delta_time):
        """ Movement and game logic """

        # Call update on all sprites (The sprites don't do much in this
        # example though.)
        if not self.done:
            self.physics_engine.update()
        # self.alien_list.update()
        # self.battleship_list.update()

        ## Player bullet update
        self.bullet_list.update()
        for bullet in self.bullet_list:
            # Check this bullet to see if it hit a coin
            hit_list = arcade.check_for_collision_with_list(bullet, self.alien_list)
            # If it did, get rid of the bullet
            if len(hit_list) > 0:
                bullet.kill()
            # For every coin we hit, add to the score and remove the coin
            for alien in hit_list:
                alien.kill()
                self.score += 1
                # Hit Sound
                # arcade.sound.play_sound(self.hit_sound)
            # If the bullet flies off-screen, remove it.
            if bullet.bottom > self.screen_height:
                bullet.kill()

        ## Alien bullet update
        aliens_to_shot = []
        if self.frame_count % 100 == 0:
            aliens_to_shot = self._select_two_shot_aliens()

        for enemy in aliens_to_shot:
            # Have a random 1 in 200 change of shooting each frame
            # print("alien started shotting ...")
            bullet = arcade.Sprite("pics/coin_01.png", SPRITE_SCALING_ALIEN_LASER * self.screen_scale)
            bullet.center_x = enemy.center_x
            bullet.center_y = enemy.center_y
            bullet.top = enemy.bottom
            bullet.angle = -90
            bullet.change_y = ALIEN_BULLET_SPEED
            self.alien_bullet_list.append(bullet)

        # Get rid of the bullet when it flies off-screen
        player_life_warning_frame = 0
        for bullet in self.alien_bullet_list:

            hit_list = arcade.check_for_collision_with_list(bullet, self.battleship_list)
            if len(hit_list) > 0:
                bullet.kill()

            if bullet.top < 0:
                bullet.kill()

            for battleship in hit_list:
                battleship.life -= 1
                player_life_warning_frame = self.frame_count
                if battleship.life == 0:
                    battleship.kill()
                    self.done = True
        self.alien_bullet_list.update()

        ## Update warning color to red when player has one life reduction
        if self.frame_count <= player_life_warning_frame + 400 and self.frame_count > 100:
            self.player_life_reduction_warning = True
        else:
            self.player_life_reduction_warning = False

        ## Update frame_count
        self.frame_count += 1

        ## Setup for the next episode
        if len(self.alien_list) == 0:
            self.episode += 1
            self._next_episode_setup(self.episode)

    def _select_two_shot_aliens(self):
        '''
        Randomly select two aliens for making shots
        :return: the sprites of the selected alien
        '''
        n_aliens = len(self.alien_list)
        if n_aliens <=2:
            return self.alien_list
        random_index = np.random.randint(n_aliens, size=2)
        return [self.alien_list[random_index[0]], self.alien_list[random_index[1]]]

    def _next_episode_setup(self, episode=1):
        # Set up aliens
        self._setup_alien()

    def _setup_alien(self):
        # Set up aliens
        alien_top_margin = int(ALIEN_TOP_MARGIN * self.screen_scale)
        alien_left_margin = int(ALIEN_LEFT_MARGIN * self.screen_scale)
        alien_width_step = int((self.screen_width - 2.0 * alien_left_margin) / ALIEN_COL_COUNT)
        alien_height_step = int(100 * self.screen_scale)
        for i in range(ALIEN_ROW_COUNT):
            # Create the alien
            for j in range(ALIEN_COL_COUNT):
                alien = arcade.Sprite("pics/alien_1.png", SPRITE_SCALING_ALIEN * self.screen_scale)

                # Position the coin
                center_x = alien_left_margin + int(1. / 2 * alien_width_step) + j * alien_width_step
                center_y = self.screen_height - (alien_top_margin + int(1. / 2 * alien_height_step)
                                            + i * alien_height_step)
                alien.center_x = center_x
                alien.center_y = center_y
                alien.boundary_left = center_x - int(ALIEN_MOVING_RANGE * self.screen_scale)
                alien.boundary_right = center_x + int(ALIEN_MOVING_RANGE * self.screen_scale)
                alien.change_x = ALIEN_SPEED
                # Add the coin to the lists
                self.alien_list.append(alien)

    def draw_update_for_state(self):
        """ Draw everything """
        # platform_event_loop.start()
        arcade.start_render()
        if not self.done:
            self.draw_game()
        else:
            self.draw_game()
            self.draw_game_over()
        arcade.finish_render()
        # self.dispatch_event()

    def _get_current_state(self):
        '''
        :return: PIL image object
        '''
        self.draw_update_for_state()
        image = arcade.draw_commands.get_image()
        return np.array(image)[:, :, :3]

    def _user_action_one_step(self, action, n_frames=1):
        '''
        :param action: action is int actions up down left right shoot idle
        :return: None
        '''
        delta_time = 0.5
        if action == 0:
            if self.battleship_sprite.center_y < self.screen_height:
                self.battleship_sprite.change_y = MOVEMENT_SPEED
            [self.update(delta_time) for _ in range(n_frames)]
            self.battleship_sprite.change_y = 0
        elif action == 1:
            if self.battleship_sprite.center_y > 0:
                self.battleship_sprite.change_y = - MOVEMENT_SPEED
            [self.update(delta_time) for _ in range(n_frames)]
            self.battleship_sprite.change_y = 0
        elif action == 2:
            if self.battleship_sprite.center_x > 0:
                self.battleship_sprite.change_x = -MOVEMENT_SPEED
            [self.update(delta_time) for _ in range(n_frames)]
            self.battleship_sprite.change_x = 0
        elif action == 3:
            if self.battleship_sprite.center_x < self.screen_width:
                self.battleship_sprite.change_x = MOVEMENT_SPEED
            [self.update(delta_time) for _ in range(n_frames)]
            self.battleship_sprite.change_x = 0
        elif action == 4:
            bullet = arcade.Sprite("pics/laserBlue01.png", SPRITE_SCALING_LASER)
            # The image points to the right, and we want it to point up. So
            # rotate it.
            bullet.angle = 90

            # Give the bullet a speed
            bullet.change_y = BULLET_SPEED

            # Position the bullet
            bullet.center_x = self.battleship_sprite.center_x
            bullet.center_y = self.battleship_sprite.center_y
            bullet.bottom = self.battleship_sprite.top

            # Add the bullet to the appropriate lists
            self.bullet_list.append(bullet)
            [self.update(delta_time) for _ in range(n_frames)]
        elif action == 5:
            [self.update(delta_time) for _ in range(n_frames)]



ACTION_MEANING = {
    0 : "UP",
    1 : "DOWN",
    2 : "LEFT",
    3 : "RIGHT",
    4 : "SHOOT",
    5 : "IDLE",
}


def main():
    """ Main method """
    window = SpaceInvader_HoustonJ2013(screen_width=200, screen_height=160)
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()
