import asyncio
from llama_index.core.workflow import StartEvent, StopEvent, Workflow, step, Context
from llama_index.core.workflow import Event
from llama_index.utils.workflow import draw_all_possible_flows
import random

class ProcessingEvent(Event):
    intermediate_result: str

class LoopEvent(Event):
    loop_output: str

class MultiStepWorkflow(Workflow):

    @step
    async def step_one(self, ev: StartEvent | LoopEvent) -> ProcessingEvent | LoopEvent:
        if random.randint(0, 1) == 0:
            print("Bad thing happened")
            return LoopEvent(loop_output="Back to step one.")
        else:
            print("Good thing happened")
            return ProcessingEvent(intermediate_result="First step complete.")

    @step
    async def step_two(self, ctx: Context, ev: ProcessingEvent) -> StopEvent:
        # Use the intermediate result
        await ctx.store.set("query", "Something to store in the context")

        final_result = f"Finished processing: {ev.intermediate_result}"
        return StopEvent(result=final_result)


async def run_workflow():
    workflow = MultiStepWorkflow(timeout=10, verbose=False)
    draw_all_possible_flows(workflow, "flow.html")
    result = await workflow.run()
    print(result)

asyncio.run(run_workflow())