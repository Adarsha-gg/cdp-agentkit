from collections.abc import Callable

import requests
from typing import Optional
from pydantic import BaseModel, Field
from cdp import Wallet
from cdp_langchain.utils.cdp_agentkit_wrapper import CdpAgentkitWrapper
from cdp_langchain.tools.cdp_tool import CdpTool

PRICE_TOOL_DESCRIPTION = """
Tool for retrieving real-time cryptocurrency token prices. 
Supports fetching current market data including price, market cap, and 24-hour price change.
Can use token symbols or contract addresses.
"""
print(CdpTool)
class GetRealDataInput(BaseModel):
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
        return {
            "price": token_data.get(f"{currency.lower()}"),
            "market_cap": token_data.get(f"{currency.lower()}_market_cap"),
            "24h_change": token_data.get(f"{currency.lower()}_24h_change"),
            "token": token_identifier
        }

    except requests.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}

# def run_token_price_tool(token_identifier: str, currency: str = "USD") -> str:
#     """Wrapper function to run the token price tool and format output.

#     Args:
#         token_identifier (str): Token symbol or contract address to fetch price for.
#         currency (str, optional): Target currency for price conversion. Defaults to "USD".

#     Returns:
#         str: Formatted string with token price information.
#     """
#     result = 
    
#     if "error" in result:
#         return result["error"]
    
#     return f"""Token Price Information for {token_identifier}:
# Price: {result['price']} {currency}
# Market Cap: {result['market_cap']} {currency}
# 24h Change: {result['24h_change']}%"""

class GetRealDataTool(CdpTool):
    """Get real data tool."""

    name: str = "get_token_price"
    description: str = PRICE_TOOL_DESCRIPTION,
    args_schema: type[BaseModel] = GetRealDataInput,
    func:Callable[..., str] =get_token_price,
    