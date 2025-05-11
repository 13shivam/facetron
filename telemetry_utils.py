import os

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor


def setup_telemetry():
    """Initialize OpenTelemetry with OTLP exporter (compatible with Jaeger)"""
    # Create a resource with service name
    resource = Resource(attributes={
        SERVICE_NAME: "faceTron-server"
    })

    # Create and configure the trace provider
    provider = TracerProvider(resource=resource)
    otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4317")

    # Configure OTLP exporter
    otlp_exporter = OTLPSpanExporter(
        endpoint=otlp_endpoint,
        insecure=True,
    )

    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))

    trace.set_tracer_provider(provider)

    # Instrument asyncio
    AsyncioInstrumentor().instrument()

    return trace.get_tracer(__name__)
