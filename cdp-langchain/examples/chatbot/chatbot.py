import os
import sys
import time

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent

# Import CDP Agentkit Langchain Extension.
from cdp_langchain.agent_toolkits import CdpToolkit
from cdp_langchain.utils import CdpAgentkitWrapper
from cdp_langchain.tools import CdpTool
import requests
from typing import Optional
from pydantic import BaseModel, Field

from chainlink.ccip import CrossChainMessenger, MessageType

CROSS_CHAIN_SWAP_PROMPT = """
Perform a cross-chain token swap using Chainlink's Cross-Chain Interoperability Protocol (CCIP).
This action enables secure token transfers and swaps between different blockchain networks.
"""

class CrossChainSwapInput(BaseModel):
    """Input schema for cross-chain swap action."""

    source_token_address: str = Field(
        ..., 
        description="Contract address of the source token to swap from",
        example="0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984"  # Uniswap token
    )
    destination_token_address: str = Field(
        ..., 
        description="Contract address of the destination token to swap to",
        example="0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48"  # USDC
    )
    amount: float = Field(
        ..., 
        description="Amount of tokens to swap",
        example=100.0
    )
    source_network_id: int = Field(
        ..., 
        description="Source blockchain network ID (Chainlink CCIP router network)",
        example=1  # Ethereum mainnet
    )
    destination_network_id: int = Field(
        ..., 
        description="Destination blockchain network ID (Chainlink CCIP router network)", 
        example=137  # Polygon
    )
    destination_wallet_address: str = Field(
        ...,
        description="Wallet address on the destination network to receive tokens",
        example="0x742d35Cc6634C0532925a3b844Bc454e4438f44e"
    )
    slippage_tolerance: Optional[float] = Field(
        default=0.01,
        description="Maximum acceptable slippage percentage",
        example=0.01  # 1% slippage
    )

def cross_chain_swap(
    wallet: Wallet, 
    source_token_address: str,
    destination_token_address: str,
    amount: float,
    source_network_id: int,
    destination_network_id: int,
    destination_wallet_address: str,
    slippage_tolerance: float = 0.01
) -> str:
    """
    Perform a cross-chain token swap using Chainlink's Cross-Chain Interoperability Protocol (CCIP).
    
    Args:
        wallet (Wallet): Source wallet for the swap
        source_token_address (str): Token contract address to swap from
        destination_token_address (str): Token contract address to swap to
        amount (float): Amount of tokens to swap
        source_network_id (int): Source blockchain network ID (Chainlink CCIP router network)
        destination_network_id (int): Destination blockchain network ID (Chainlink CCIP router network)
        destination_wallet_address (str): Wallet address on destination network to receive tokens
        slippage_tolerance (float, optional): Maximum acceptable price slippage. Defaults to 0.01 (1%).
    
    Returns:
        str: Cross-chain swap transaction details and confirmation
    """
    try:
        # Initialize Chainlink Cross-Chain Messenger
        ccip_messenger = CrossChainMessenger(
            source_network_id=source_network_id,
            destination_network_id=destination_network_id
        )

        # Fetch current swap quote and validate
        swap_quote = ccip_messenger.get_quote(
            source_token=source_token_address,
            destination_token=destination_token_address,
            amount=amount,
            slippage_tolerance=slippage_tolerance
        )

        # Approve token spending for CCIP router
        wallet.approve_token_spending(
            token_address=source_token_address, 
            amount=amount,
            spender=ccip_messenger.router_address
        )

        # Execute cross-chain transfer
        ccip_message = ccip_messenger.send_message(
            receiver_address=destination_wallet_address,
            message_type=MessageType.TRANSFER_TOKEN,
            payload={
                "token_in": source_token_address,
                "token_out": destination_token_address,
                "amount": amount
            }
        )

        # Wait for message confirmation
        message_status = ccip_message.wait_for_confirmation()

        return f"""
        Chainlink CCIP Cross-Chain Swap Completed:
        - From: {source_token_address} (Network {source_network_id})
        - To: {destination_token_address} (Network {destination_network_id})
        - Amount: {amount}
        - Destination Wallet: {destination_wallet_address}
        - CCIP Message ID: {ccip_message.message_id}
        - Transaction Status: {message_status}
        """
    
    except Exception as e:
        logging.error(f"Cross-chain swap failed: {str(e)}")
        return f"Cross-chain swap failed: {str(e)}"

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
    try:
        # Determine if input is a contract address or a token symbol
        if token_identifier.startswith('0x'):
            # Use contract address lookup
            url = f"https://api.coingecko.com/CG-2pUQLuvdJhQVuWUuyGZ61kRS/v3/simple/token_price/ethereum?contract_addresses={token_identifier}&vs_currencies=USD&include_market_cap=true&include_24hr_change=true"
        else:
            # Use token symbol lookup
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={token_identifier}&vs_currencies=USD&include_market_cap=true&x_cg_demo_api_key=CG-2pUQLuvdJhQVuWUuyGZ61kRS"

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

    # Ensure tools is a list and add the new tool
    if tools is None:
        tools = []
    tools.append(realTool)

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


# Mode Selection
def choose_mode():
    """Choose whether to run in autonomous or chat mode based on user input."""
    while True:
        print("\nAvailable modes:")
        print("1. chat    - Interactive chat mode")
        print("2. auto    - Autonomous action mode")

        choice = input("\nChoose a mode (enter number or name): ").lower().strip()
        if choice in ["1", "chat"]:
            return "chat"
        elif choice in ["2", "auto"]:
            return "auto"
        print("Invalid choice. Please try again.")


def main():
    """Start the chatbot agent."""
    agent_executor, config = initialize_agent()

    mode = choose_mode()
    if mode == "chat":
        run_chat_mode(agent_executor=agent_executor, config=config)
    elif mode == "auto":
        run_autonomous_mode(agent_executor=agent_executor, config=config)


if __name__ == "__main__":
    print("Starting Agent...")
    main()
