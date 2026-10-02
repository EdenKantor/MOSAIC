from mosaic.core.config import TeacherAccess
from mosaic.core.models import ModelRequest, ModelResponse
from mosaic.providers.base import ModelProvider


class TeacherUnavailable(RuntimeError):
    pass


class TeacherCapability:
    """The only runtime route to teacher inference; withdrawal drops the provider reference."""

    def __init__(
        self, provider: ModelProvider, access: TeacherAccess = TeacherAccess.AVAILABLE
    ) -> None:
        self.__provider: ModelProvider | None = (
            provider if access == TeacherAccess.AVAILABLE else None
        )
        self.__access = access
        self.provider_implementation = f"{type(provider).__module__}.{type(provider).__qualname__}"

    @property
    def access(self) -> TeacherAccess:
        return self.__access

    def withdraw(self) -> None:
        self.__provider = None
        self.__access = TeacherAccess.WITHDRAWN

    def require_available(self) -> None:
        if self.__access != TeacherAccess.AVAILABLE or self.__provider is None:
            raise TeacherUnavailable("Teacher access is withdrawn")

    async def generate(self, request: ModelRequest) -> ModelResponse:
        self.require_available()
        provider = self.__provider
        assert provider is not None
        return await provider.generate(request)
