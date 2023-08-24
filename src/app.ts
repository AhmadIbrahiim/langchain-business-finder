//Import the OpenAPI Large Language Model (you can import other models here eg. Cohere)
import { OpenAI } from "langchain/llms/openai"
import { z } from "zod"

//Import the agent executor module
import { initializeAgentExecutorWithOptions } from "langchain/agents"

//Import the SerpAPI and Calculator tools
import { SerpAPI } from "langchain/tools"
import { Calculator } from "langchain/tools/calculator"

//Load environment variables (populate process.env from .env file)
import * as dotenv from "dotenv"
import { PromptTemplate } from "langchain/prompts"
import { OutputFixingParser, StructuredOutputParser } from "langchain/output_parsers"
dotenv.config()

export const run = async () => {
  //Instantiante the OpenAI model
  //Pass the "temperature" parameter which controls the RANDOMNESS of the model's output. A lower temperature will result in more predictable output, while a higher temperature will result in more random output. The temperature parameter is set between 0 and 1, with 0 being the most predictable and 1 being the most random
  const model = new OpenAI({ temperature: 0 })

  //
  const parser = StructuredOutputParser.fromZodSchema(
    z.object({
      company: z.string().describe("Company name"),
      phone: z.string().describe("Company phone number"),
      location: z.string().describe("Company location"),
    }),
  )

  const formatInstructions = parser.getFormatInstructions()
  //Create a list of the instatiated tools
  const tools = [
    new SerpAPI(process.env.SERP_API_KEY, {
      location: "United States",
    }),
    new Calculator(),
  ]

  //Construct an agent from an LLM and a list of tools
  //"zero-shot-react-description" tells the agent to use the ReAct framework to determine which tool to use. The ReAct framework determines which tool to use based solely on the tool’s description. Any number of tools can be provided. This agent requires that a description is provided for each tool.
  const executor = await initializeAgentExecutorWithOptions(tools, model, {
    agentType: "zero-shot-react-description",
    verbose: false,
  })

  // Prompt to search for company phone number and name and location on google
  const prompt = new PromptTemplate({
    template: "Extract company name , phone number and location from google search results for {company}",
    inputVariables: ["company"],
    partialVariables: { format_instructions: formatInstructions },
  })

  //Run the agent
  const result = await executor.call({
    input: await prompt.format({ company: "goodcall.ai" }),
  })

  console.log(result.output)
  console.log("#".repeat(80))
  try {
    console.log(await parser.parse(result.output))
  } catch (e) {
    console.error("Failed to parse bad output: ", e)

    const fixParser = OutputFixingParser.fromLLM(model, parser)
    const output = await fixParser.parse(result.output)

    console.log(output)
    console.log("^".repeat(80))
  } // parse excutor result
}

run()
