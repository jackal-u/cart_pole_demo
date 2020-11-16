import gym


from dqn import Agent


def start():
    env = gym.make('CartPole-v0')

    params = {
        'gamma': 0.8,
        'epsi_high': 0.9,
        'epsi_low': 0.05,
        'decay': 500,
        'lr': 0.001,
        'capacity': 10000,
        'batch_size': 64,
        'state_space_dim': env.observation_space.shape[0],
        'action_space_dim': env.action_space.n
    }
    agent = Agent(**params)

    score = []
    mean = []

    for episode in range(1000):
        s0 = env.reset()
        total_reward = 1
        for i in range(200):
            env.render()
            a0 = agent.act(s0)
            s1, r1, done, _ = env.step(a0)

            if done:
                r1 = -1

            agent.put(s0, a0, r1, s1)

            if done:
                break

            total_reward += r1
            s0 = s1
            agent.learn()

        score.append(total_reward)
        mean.append(sum(score[-100:]) / 100)
        print(total_reward)
if __name__ == '__main__':
    start()
