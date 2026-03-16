from django.urls import path
from .views import GetGroqLLMResponseView

urlpatterns = [
    path("prompt/", GetGroqLLMResponseView.as_view()),
    path("prompt", GetGroqLLMResponseView.as_view()),
]