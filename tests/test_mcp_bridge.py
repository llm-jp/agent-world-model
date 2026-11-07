"""The bridge must reproduce what fastapi-mcp 0.4.0 on mcp 1.26 served.

The expected values below were recorded from fastapi-mcp 0.4.0 with mcp 1.26.0
and fastapi 0.115.12 serving the same app.
"""

import asyncio

import pytest
from fastapi import FastAPI, HTTPException, Path
from fastapi.testclient import TestClient
from mcp import Client
from pydantic import BaseModel, Field

from awm.core.mcp_bridge import mount_mcp
from awm.core.mcp_bridge.server import create_mcp_server


class Product(BaseModel):
    id: int = Field(..., description="Product ID")
    name: str = Field(..., description="商品名")
    category: str | None = Field(None, description="Category, if any")
    price: float
    stock: int


class ProductCreate(BaseModel):
    name: str = Field(..., min_length=1, description="商品名")
    price: float = Field(..., gt=0)


def make_app() -> FastAPI:
    app = FastAPI(title="Sample Shop API")
    products = {
        1: Product(id=1, name="りんご", category="fruit", price=120.0, stock=10)
    }

    @app.get(
        "/products/{product_id}",
        response_model=Product,
        summary="Get a product",
        operation_id="get_product",
    )
    def get_product(product_id: int = Path(..., description="ID of the product")):
        if product_id not in products:
            raise HTTPException(status_code=404, detail="Product not found")
        return products[product_id]

    @app.post(
        "/products",
        response_model=Product,
        status_code=201,
        summary="Create a product",
        operation_id="create_product",
    )
    def create_product(payload: ProductCreate):
        product = Product(id=2, name=payload.name, price=payload.price, stock=0)
        products[2] = product
        return product

    return app


GET_PRODUCT_TOOL = {
    "name": "get_product",
    "description": 'Get a product\n\n### Responses:\n\n**200**: Successful Response (Success Response)\nContent-Type: application/json\n\n**Example Response:**\n```json\n{\n  "id": 1,\n  "name": "Name",\n  "price": 1.0,\n  "stock": 1\n}\n```',
    "inputSchema": {
        "type": "object",
        "properties": {
            "product_id": {
                "type": "integer",
                "description": "ID of the product",
                "title": "product_id",
            }
        },
        "title": "get_productArguments",
        "required": ["product_id"],
    },
}


async def _call(name: str, arguments: dict) -> tuple[bool, str]:
    async with Client(create_mcp_server(make_app())) as client:
        result = await client.call_tool(name, arguments)
    return result.is_error, result.content[0].text


def test_tools_match_fastapi_mcp() -> None:
    async def list_tools():
        async with Client(create_mcp_server(make_app())) as client:
            return (await client.list_tools()).tools

    tools = {t.name: t for t in asyncio.run(list_tools())}
    tool = tools["get_product"]
    assert {
        "name": tool.name,
        "description": tool.description,
        "inputSchema": tool.input_schema,
    } == GET_PRODUCT_TOOL
    assert list(tool.input_schema) == list(GET_PRODUCT_TOOL["inputSchema"])
    assert set(tools) == {"get_product", "create_product"}


def test_success_returns_indented_json() -> None:
    assert asyncio.run(_call("get_product", {"product_id": 1})) == (
        False,
        '{\n  "id": 1,\n  "name": "りんご",\n  "category": "fruit",\n  "price": 120.0,\n  "stock": 10\n}',
    )


def test_error_texts_match_mcp_1x() -> None:
    cases = {
        ("get_product", (("product_id", 999),)): (
            'Error calling get_product. Status code: 404. Response: {"detail":"Product not found"}'
        ),
        (
            "get_product",
            (),
        ): "Input validation error: 'product_id' is a required property",
        ("get_product", (("product_id", "abc"),)): (
            "Input validation error: 'abc' is not of type 'integer'"
        ),
        ("create_product", (("name", "x"), ("price", -1))): (
            "Input validation error: -1 is less than or equal to the minimum of 0.0"
        ),
        ("no_such_tool", (("x", 1),)): "Unknown tool: no_such_tool",
    }
    for (name, args), expected in cases.items():
        assert asyncio.run(_call(name, dict(args))) == (True, expected)


# generated environments still register startup handlers with on_event
@pytest.mark.filterwarnings("ignore:\\s*on_event is deprecated:DeprecationWarning")
def test_mount_keeps_startup_handlers() -> None:
    app = make_app()
    started = []

    @app.on_event("startup")
    def on_startup():
        started.append(True)

    mount_mcp(app)
    with TestClient(app) as client:
        response = client.post(
            "/mcp",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "test", "version": "0"},
                },
            },
            headers={"Accept": "application/json, text/event-stream"},
        )
    assert response.status_code == 200
    assert response.json()["result"]["serverInfo"]["name"] == "Sample Shop API"
    assert started == [True]
