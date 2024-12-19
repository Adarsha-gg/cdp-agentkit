import os
import sys
import time
import requests
import base64
import speech_recognition as sr
import threading
import logging
import re

from dotenv import load_dotenv
from typing import Optional,Dict, Any
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Import CDP Agentkit Langchain Extension.
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from cdp_langchain.tools import CdpTool

from cdp import Wallet
from pydantic import BaseModel, Field
import qrcode
from qrcode.main import QRCode
from io import StringIO

GENERATE_QR_PROMPT = """
This tool generates a QR code for an MPC wallet address and displays it in the terminal. This is useful for quickly sharing wallet addresses for receiving assets or connecting to dApps."""


class GenerateQrInput(BaseModel):
    """Input argument schema for QR code generation action."""
    
    wallet_address: str = Field(
        ...,
        description="The wallet address to encode in the QR code",
        example="0x036CbD53842c5426634e7929541eC2318f3dCF7e"
    )
    
    network: str = Field(
        ...,
        description="The network ID for the wallet address (e.g., 'ETHEREUM_MAINNET', 'POLYGON_MAINNET', 'BINANCE_SMART_CHAIN_MAINNET')",
        example="ETHEREUM_MAINNET"
    )

def generate_qr(wallet: Wallet, wallet_address: str, network: str) -> str:
    """Generate a QR code for a wallet address and display it in the terminal.
    
    Args:
        wallet (Wallet): The wallet instance (used for validation).
        wallet_address (str): The wallet address to encode in the QR code.
        
    Returns:
        str: ASCII representation of the QR code along with the encoded address.
    """
    # Create QR code instance
    qr = QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=1,
        border=1,
    )
    
    # Add data
    qr.add_data(f"{network}:{wallet_address}")
    qr.make(fit=True)
    
    # Create string buffer to capture ASCII output
    f = StringIO()
    qr.print_ascii(out=f)
    f.seek(0)
    
    # Get ASCII QR code
    qr_ascii = f.read()
    
    return f"QR Code for wallet address {wallet_address}:\n\n{qr_ascii}"

class VoiceCommandHandler:
    def __init__(self, agent_executor, config):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.is_listening = False
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.WARNING)  # Suppress detailed HTTP logs
        self.agent_executor = agent_executor
        self.config = config

    def process_voice_command(self, command):
        """
        Process the recognized voice command using the agent executor.

        Args:
            command (str): Voice command to process.

        Returns:
            str: Agent's response to the command.
        """
        try:
            response = ""
            for chunk in self.agent_executor.stream(
                 {"messages": [HumanMessage(content=command)]},
                self.config
            ):
                if "agent" in chunk:
                    response += chunk["agent"]["messages"][0].content
                elif "tools" in chunk:
                    response += chunk["tools"]["messages"][0].content
                 

            return self.format_response(response)
        except Exception as e:
            error_msg = f"Error processing command: {e}"
            self.logger.error(error_msg)
            return error_msg

    def format_response(self, response):
        """
        Format the agent's response for better readability.

        Args:
            response (str): Raw response from the agent.

        Returns:
            str: Formatted response.
        """
        return "\n".join(line.strip() for line in response.split("\n") if line.strip())

    def listen_and_respond(self):
        """
        Listen for voice commands and process them.
        """
        self.is_listening = True
        print("Voice mode activated. Say 'exit' to end.")

        while self.is_listening:
            try:
                with self.microphone as source:
                    self.logger.info("Listening...")
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio = self.recognizer.listen(source, timeout=15, phrase_time_limit=15)

                try:
                    # Recognize speech
                    command = self.recognizer.recognize_google(audio).lower()
                    print(f"Command recognized: {command}")
                    print("-------------------")   
                    # Check for exit command
                    if command == 'exit':
                        print("Exiting voice mode. Goodbye!")
                        self.is_listening = False
                        break

                    # Process the command
                    response = self.process_voice_command(command)

                    # Print the response
                    print(f"\n{response}\n")
                    print("-------------------")   

                except sr.UnknownValueError:
                    print("Sorry, I could not understand that. Could you please repeat?")
                except sr.RequestError:
                    print("Speech recognition service is unavailable. Please try again later.")

            except Exception as e:
                self.logger.error(f"Unexpected error in voice processing: {e}")
                print("An unexpected error occurred. Please try again.")

    def start(self):
        """
        Start the voice command listener in a separate thread.
        """
        listener_thread = threading.Thread(target=self.listen_and_respond, daemon=True)
        listener_thread.start()
        return listener_thread

    def stop(self):
        """
        Stop the voice command listener.
        """
        self.is_listening = False
        print("Voice command listener stopped.")



load_dotenv()

WALLET_SEARCH_PROMPT = """
This tool searches for all transactions (trades, swaps, transfers) associated with a given wallet address. It retrieves a summary of recent transaction activities across different protocols and token types.
"""

class WalletSearchInput(BaseModel):
    """Input argument schema for Etherscan wallet transaction search."""

    wallet_address: str = Field(
        ...,
        description="The Ethereum wallet address to search for transactions (e.g., '0x742d35Cc6634C0532925a3b844Bc454e4438f44e')",
        example="0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
    )
    max_transactions: int = Field(
        default=50,
        description="Maximum number of transactions to retrieve",
        ge=1,
        le=1000
    )

def search_wallet_transactions(wallet_address: str, max_transactions: int = 50) -> str:
    """
    Search for transactions associated with a specific wallet address.

    Args:
        wallet_address (str): Ethereum wallet address to search.
        api_key (str):  API key.
        max_transactions (int, optional): Maximum number of transactions to retrieve. Defaults to 50.

    Returns:
        str: Summarized transaction information.
    """
    # Etherscan API endpoints

    API_KEY = os.getenv('WALLET_API_KEY')
    encoded_key = base64.b64encode(API_KEY.encode()).decode()

    query = """
    query providerPorfolioQuery($addresses: [Address!]!, $networks: [Network!]!) {
    portfolio(addresses: $addresses, networks: $networks) {
        tokenBalances {
        address
        network
        token {
            balance
            balanceUSD
            baseToken {
            name
            symbol
            }
        }
        }
    }
    }
    """
    try:
        response = requests.post(
            'https://public.zapper.xyz/graphql',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Basic {encoded_key}'
            },
            json={
                'query': query,
                'variables': {
                    'addresses': [wallet_address],
                    'networks': ['ETHEREUM_MAINNET','POLYGON_MAINNET','BINANCE_SMART_CHAIN_MAINNET']
                }
            },
            timeout=30
        )

        response.raise_for_status()
        data = response.json()

        if 'errors' in data:
            raise ValueError(f"GraphQL Errors: {data['errors']}")

        balance = data['data']['portfolio']['tokenBalances']
        result = ''
        for item in balance:
            result += f"""Network:" {item["network"]}
            Symbol: {item["token"]["baseToken"]["symbol"]}
            Balance: {item["token"]["balance"]}
            Balance USD: {item["token"]["balanceUSD"]}\n"""

        return result    

    except requests.RequestException as e:
        print(f"Request failed: {e}")
        raise
    except ValueError as e:
        print(f"Data validation failed: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

PRICE_TOOL_DESCRIPTION = """
Tool for retrieving real-time cryptocurrency token prices. 
Supports fetching current market data including price, market cap, and 24-hour price change.
Can use token symbols or contract addresses.
"""

class TokenPriceTool(BaseModel):
    """Pydantic model for token price retrieval tool."""
    token_identifier: str = Field(
        ..., 
        description="Token name (e.g., 'Bitcoin', 'Ethereum') or contract address",
        examples=["Bitcoin", "Ethereum", "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984"]
    )
    currency: Optional[str] = Field(
        default="USD", 
        description="Target currency for price conversion",
        examples=["USD", "EUR", "GBP"]
    )

def get_token_price(token_identifier: str, currency: str = "USD") -> dict:
    """Retrieve real-time price information for a cryptocurrency token.
    Args:
        token_identifier (str): Token symbol or contract address to fetch price for.
        currency (str, optional): Target currency for price conversion. Defaults to "USD".
    Returns:
        dict: A dictionary containing token price information.
    """
    Api = os.getenv('COINGECKO_API_KEY')
    try:
        # Determine if input is a contract address or a token symbol
        if token_identifier.startswith('0x'):
            # Use contract address lookup
            url = f"https://api.coingecko.com/api/v3/simple/token_price/id={token_identifier}include_market_cap=true&include_24hr_vol=true&include_24hr_change=true&x_cg_demo_api_key={Api}"
        else:
            # Use token symbol lookup
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={token_identifier}&vs_currencies=USD&include_market_cap=true&x_cg_demo_api_key={Api}"
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        # Process and format the response
        if not data:
            return {"error": "Token not found"}
        # Extract price information
        token_data = list(data.values())[0] if data else {}
        result = f"""" Details for {token_identifier}:
            Price: {token_data.get(f"{currency.lower()}")} {currency}
            Market Cap: {token_data.get(f"{currency.lower()}_market_cap")} {currency}
                    """
        return result
    except requests.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}

# Configure a file to persist the agent's CDP MPC Wallet Data.
wallet_data_file = "wallet_data.txt"


def initialize_agent():
    """Initialize the agent with CDP Agentkit."""
    # Initialize LLM.
    llm = ChatOpenAI(model="gpt-4o-mini")

    wallet_data = None

    if os.path.exists(wallet_data_file):
        with open(wallet_data_file) as f:
            wallet_data = f.read()

    # Configure CDP Agentkit Langchain Extension.
    values = {}
    if wallet_data is not None:
        # If there is a persisted agentic wallet, load it and pass to the CDP Agentkit Wrapper.
        values = {"cdp_wallet_data": wallet_data}

    agentkit = CdpAgentkitWrapper(**values)

    # persist the agent's CDP MPC Wallet Data.
    wallet_data = agentkit.export_wallet()
    with open(wallet_data_file, "w") as f:
        f.write(wallet_data)

    # Initialize CDP Agentkit Toolkit and get tools.
    cdp_toolkit = CdpToolkit.from_cdp_agentkit_wrapper(agentkit)
    tools = cdp_toolkit.get_tools()

    realTool = CdpTool(
        name="token_price",
        description=PRICE_TOOL_DESCRIPTION,
        cdp_agentkit_wrapper=agentkit,
        func=get_token_price,
        args_schema=TokenPriceTool,
    )

    visualizer = CdpTool(
        name="visualize_wallet",
        description=WALLET_SEARCH_PROMPT,
        cdp_agentkit_wrapper=agentkit,
        func=search_wallet_transactions,
        args_schema=WalletSearchInput,
    )

    qr_code = CdpTool(
        name="qr_code",
        description=GENERATE_QR_PROMPT,
        cdp_agentkit_wrapper=agentkit,
        func=generate_qr,
        args_schema=GenerateQrInput,
    )

    # Ensure tools is a list and add the new tool
    if tools is None:
        tools = []

    tools.append(realTool)
    tools.append(visualizer)
    tools.append(qr_code)

    # Store buffered conversation history in memory.
    memory = MemorySaver()
    config = {"configurable": {"thread_id": "CDP Agentkit Chatbot Example!"}}

    
    # Create ReAct Agent using the LLM and CDP Agentkit tools.
    return create_react_agent(
        llm,
        tools=tools,
        checkpointer=memory,
        state_modifier=(
            "You are a helpful agent that can interact onchain using the Coinbase Developer Platform AgentKit. "
            "You are empowered to interact onchain using your tools. If you ever need funds, you can request "
            "them from the faucet if you are on network ID 'base-sepolia'. If not, you can provide your wallet "
            "details and request funds from the user. Before executing your first action, get the wallet details "
            "to see what network you're on. If there is a 5XX (internal) HTTP error code, ask the user to try "
            "again later. If someone asks you to do something you can't do with your currently available tools, "
            "you must say so, and encourage them to implement it themselves using the CDP SDK + Agentkit, "
            "recommend they go to docs.cdp.coinbase.com for more information. Be concise and helpful with your "
            "responses. Refrain from restating your tools' descriptions unless it is explicitly requested."
            
        ),

    ), config


# Autonomous Mode
def run_autonomous_mode(agent_executor, config, interval=10):
    """Run the agent autonomously with specified intervals."""
    print("Starting autonomous mode...")
    while True:
        try:
            # Provide instructions autonomously
            thought = (
                "Be creative and do something interesting on the blockchain. "
                "Choose an action or set of actions and execute it that highlights your abilities."
            )

            # Run agent in autonomous mode
            for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content=thought)]}, config
            ):
                if "agent" in chunk:
                    print(chunk["agent"]["messages"][0].content)
                elif "tools" in chunk:
                    print(chunk["tools"]["messages"][0].content)
                print("-------------------")

            # Wait before the next action
            time.sleep(interval)

        except KeyboardInterrupt:
            print("Goodbye Agent!")
            sys.exit(0)


# Chat Mode
def run_chat_mode(agent_executor, config):

    """Run the agent interactively based on user input."""
    print("Starting chat mode... Type 'exit' to end.")
    while True:
        try:
            user_input = input("\nPrompt: ")
            if user_input.lower() == "exit":
                break

            # Run agent with the user's input in chat mode
            for chunk in agent_executor.stream(
                {"messages": [HumanMessage(content=user_input)]}, config
            ):
                if "agent" in chunk:
                    print(chunk["agent"]["messages"][0].content)
                elif "tools" in chunk:
                    print(chunk["tools"]["messages"][0].content)
                print("-------------------")

        except KeyboardInterrupt:
            print("Goodbye Agent!")
            sys.exit(0)

# Voice mode
def run_voice_mode(agent_executor, config):
    """
    Run the agent interactively based on voice input.
    
    Args:
        agent_executor: Agent executor instance.
        config: Agent configuration.
    """
    try:
        voice_handler = VoiceCommandHandler(agent_executor, config)
        voice_thread = voice_handler.start()

        # Keep the main thread running until voice thread completes
        voice_thread.join()

    except KeyboardInterrupt:
        print("Voice mode interrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"Error in voice mode: {e}")
        sys.exit(1)



# Mode Selection
def choose_mode():
    """Choose whether to run in autonomous or chat mode based on user input."""
    while True:
        print("\nAvailable modes:")
        print("1. chat    - Interactive chat mode")
        print("2. auto    - Autonomous action mode")
        print("3. voice    - Interactive voice mode")

        choice = input("\nChoose a mode (enter number or name): ").lower().strip()
        if choice in ["1", "chat"]:
            return "chat"
        elif choice in ["2", "auto"]:
            return "auto"
        elif choice in ["3", "voice"]:
            return "voice"   
        print("Invalid choice. Please try again.")


def main():
    """Start the chatbot agent."""
    agent_executor, config = initialize_agent()
    
    mode = choose_mode()
    if mode == "chat":
        run_chat_mode(agent_executor=agent_executor, config=config)
    elif mode == "auto":
        run_autonomous_mode(agent_executor=agent_executor, config=config)
    elif mode == "voice":
        run_voice_mode(agent_executor=agent_executor, config=config)    


if __name__ == "__main__":
    print("Starting Agent...")
    main()