# What is WHALE?

- Multiagent Harness
- With looping capabilities
- With Ontologies and Knowledge Base 


## Glossary 

*Agent AI Harness* - runtime for multiagent systems with orchestrations based on LLM backbone.
*Loop Engineering*  - a method to work with AI agents in a loop controlled way. Following the pattern: act → observe → decide → repeat with exit criteria
*Swarm* - multiple agents working in parallel, executing a piece of instruction from the system or the user



## Specs
- extensive memory, ideally it should have structured memory
- Should be able to launch swarms of agents
- Should have stop criteria
- It should be self-learning!
- Should have tools available, as many as possible
- Looping



## Architecture
(
1. pick up (remember what was before aka retrieve from memory)
2. learn (run a loop to have high level understanding what was before)
3. plan (run a loop to iterate over planning, this includes spinning up the agents to plan (a swarm), maybe! plan with tree like structure, tree-like graph  ) 
4. research ( research further down to strengthen the plan, also with the swarm of agentic) 
5. Execute the plan -> produce the output for the user, via swarm. 
6. Register the completion fact, save the output, match it to the plan from step 3.
7. Decide -> Provide the output
    |
   GO LOOP OVER
) -> LOOP OVER


### Architectural aspects
- Multiagent
- Memory (semistructure), llm wiki type, Semantica  !
- Learning capabilities
- powerful on simple fast cheap LLM backbones
- Swarm capabilities ! 
- rhizomatic-like structure !

LOOP around with success criteria



### Worth mentioning
System Prompts
capability surface (skills, plugins, MCP, tools, platforms) 
Autodiscovery of tools, autocreation of skills etc (via hooks)
Bundled Infrastructure (filesystem, sandbox, browser)
Orchestration Logic (subagent spawning, handoffs, model routing)
Hooks/Middleware for deterministic execution (compaction, continuation, lint checks)
Agents and task survival `cronjob` or `terminal(background=True, notify_on_complete=True)`


## Form
- CLI - bundles
- UI Too


## Dev Tools
- VS Code
- Python + Typer OR Go + Cobra
- github
- github actions for CI/CD




- opencode + MiniMax M3
- kilocode
- Codex
- Claude Code
and other


## Design

CLI Tool


`whale`  continues session from the pwd  / execute a prompt
Available commands:

-h --help
-m --models / LLM backbone
-s --sessions
-sk --skills add nameoftheskill/SKILL.md
-ag --agents add [AGENT]
-aa --apps add type: OAuth 
-mm --memory manage [forget | relearn | add ]

-cc --custom commands
--c -add  


## Config 
System folder + settings
json file (?)
~/.whale
skills
agents
-default
-custom
mcps
tool (loaded dynamically)
settings.json
apps
- oauth.json
- specs.json (can have OpenAPI)

memory -> mongo like, sql like (with a native tools to read from it)

sessions 
SessionId , MessageId, tools_called, content



## Important things
- Cheap to run
- Vast access to file system and tools
- Should complete the tasks efficiently
- Should have feedback loop



## Acceptance criteria
- Reliable harness with memory and self-evolving capabilities
- Lean in token consumption
- Guaranteed result to execute tasks and deliver


## Misc
landing page
wiki docs
native integrations: memory engines like openwolf
curl and brew installation, apt etc installations



## Further reading
https://github.com/sturlese/stigmergy/blob/main/docs/decisions/023-learning-loop.md
https://github.com/semantica-agi/semantica/blob/main/ARCHITECTURE.md



## Current progress:
20.08.2026:
- Implemented: architectural research
- Memory hooks, session init

Next steps: proper memory arrangement, ideas: always ask llm to return a prepared graph objects like node or branch (as we intend to have a memory of graph structure) and save it immediately. Or we fire a tool call with fire-and-forget option to form this graph primitive.

28.08.2026
- Looked at cognee, neo4j agent memory, semantica and openwolf
- semantica is data lake and ETL pipeline exposed to the agents with graph storage capabilities
- Neo4j agent is a graph like , ER memory interface, opinionated and dependent on the backbones
- Cognee is the closest competitor to whale-harness's memory ambition in the open-source world: a pluggable, persistent, multi-layered memory engine for AI agents that combines a knowledge graph + vector store + session cache + (optional) LLM-driven self-improvement loops. It is the only reviewed engine that takes both structured (graph) memory and feedback-driven self-tuning seriously.
- openwolf is a simplistic coding memory with hooks




# Useful R n D prompts

Please research the git@github.com:cytostack/openwolf.git , clone it in tmp directory , run tree, grep and other read commands. and please write how it works here docs/memory engines overview/docs/memory engines overview/openwolf_memory.md . Also give your comments if it fits to be a memory agent for our whale Harness. Does it has self learning capabilities?