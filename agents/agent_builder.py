


def creator_agent(config: dict, store=None):
    from agents.tools.common_tools import create_agent, save_agent_result_to_redis, to_jsonable, redis_client
    return create_agent(config, store, save_agent_result_to_redis, to_jsonable, redis_client)