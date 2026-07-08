"""ノード間通信（/advertise, /probe, /dispatch）で用いるJSONメッセージのpydanticスキーマを定義する．"""

from pydantic import BaseModel, Field


class AdvertiseRequest(BaseModel):
    """自ノードの現在状態をピアに周知するハートビートのリクエスト．"""

    node_id: str
    domain: str
    domain_embedding: list[float]
    load: float = Field(ge=0.0, le=1.0)
    timestamp: int = Field(description="UNIX epoch秒")


class AdvertiseResponse(BaseModel):
    """advertiseの受理応答．"""

    status: str = "ok"


class ProbeRequest(BaseModel):
    """依頼者から各ノードへ担当可否のみを問い合わせるリクエスト．"""

    request_id: str
    query_summary: str
    query_embedding: list[float]
    from_: str = Field(alias="from", description="依頼元ノードのnode_id")

    model_config = {"populate_by_name": True}


class ProbeResponse(BaseModel):
    """担当可否問い合わせへの応答．confidenceは0〜1の自己申告スコア．"""

    request_id: str
    node_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    estimated_latency_ms: int


class DispatchRequest(BaseModel):
    """選定された専門家へ本文の回答生成を依頼するリクエスト．"""

    request_id: str
    full_query: str


class DispatchResponse(BaseModel):
    """回答生成の結果．"""

    request_id: str
    node_id: str
    answer_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    gen_time_ms: int


class ErrorResponse(BaseModel):
    """全エンドポイント共通のエラー応答（400/503/504で使用）．"""

    error: str
