from django.urls import path
from .views import GetOllamaLLMResponseView

urlpatterns = [
    path("<str:model_name>/prompt", GetOllamaLLMResponseView.as_view()),
    path("<str:model_name>/prompt/", GetOllamaLLMResponseView.as_view())
]

