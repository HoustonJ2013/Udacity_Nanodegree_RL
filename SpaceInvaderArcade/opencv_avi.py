import cv2
import numpy as np

def record_policy(env, video_file="video.avi", Policy="Random", max_length=1000, fps=20, frame_skip=5):
    '''
    :param Env: OpenAI gym enviroment
    :param Policy: A trained policy
    :return: the video played by policy
    '''
    if Policy == "Random":
        game_over = False
        total_reward = 0
        step = 0
        while (not game_over) and step <= max_length:
            action = env.action_space.sample() ## Return a random action in action space: INT
            for _ in range(frame_skip):
                state, reward, game_over, _ = env.step(action)
                total_reward += reward
                if step == 0:
                    height, width, layers = state.shape
                    video = cv2.VideoWriter(video_file, 0, fps, (width, height))
                    video.write(state)
                    step += 1
                else:
                    video.write(state)
                    step += 1
    else:
        pass
    video.release()


## Test function for debugging purpose
def test_generate_images(env, folder="./pics/", Policy="Random", max_length=1000):
    '''
    :param Env: OpenAI gym enviroment
    :param Policy: A trained policy
    :return: the video played by policy
    '''
    if Policy == "Random":
        game_over = False
        total_reward = 0
        step = 0
        while (not game_over) and step <= max_length:
            action = env.action_space.sample() ## Return a random action in action space: INT
            state, reward, game_over, _ = env.step(action)
            total_reward += reward
            if step == 0:
                height, width, layers = state.shape
                print(height, width, layers)
                cv2.imwrite(folder + "%4i.png"%(step), state)
                step += 1
            else:
                cv2.imwrite(folder + "%4i.png"%(step), state)
                step += 1
    else:
        pass
