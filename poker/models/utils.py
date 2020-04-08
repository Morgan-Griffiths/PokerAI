from agents.agent import Agent

def return_agent(agent_type,nS,nO,nA,seed,params):
    if agent_type == 'baseline':
        agent = Agent(nS,nO,nA,seed,params)
    elif agent_type == 'dqn':
        agent = Priority_DQN(nS,nO,nA,seed,params)
    else:
        raise ValueError(f'Agent not supported {agent_type}')
    return agent