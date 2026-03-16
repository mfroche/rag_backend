from django.urls import path
from .views import DailyPatientFoodIntakeRecommenderView, WeeklyPatientFoodIntakeRecommenderView, MonthlyPatientFoodIntakeRecommenderView, GeneralPatientFoodIntakeRecommenderView

# BASE ENDPOINT: /api/recommend/

urlpatterns = [
    # Patient Daily food intake recommendations
    path("patient/<int:pk>/daily", DailyPatientFoodIntakeRecommenderView.as_view()),
    path("patient/<int:pk>/daily/", DailyPatientFoodIntakeRecommenderView.as_view()),

    # Patient Weekly food intake recommendations
    path("patient/<int:pk>/weekly", WeeklyPatientFoodIntakeRecommenderView.as_view()),
    path("patient/<int:pk>/weekly/", WeeklyPatientFoodIntakeRecommenderView.as_view()),

    # Patient Monthly food intake recommendations
    path("patient/<int:pk>/monthly", MonthlyPatientFoodIntakeRecommenderView.as_view()),
    path("patient/<int:pk>/monthly/", MonthlyPatientFoodIntakeRecommenderView.as_view()),


    # Genaral Patient recommendations for macros and micros
    path("patient/<int:pk>/general", GeneralPatientFoodIntakeRecommenderView.as_view()),
    path("patient/<int:pk>/general/", GeneralPatientFoodIntakeRecommenderView.as_view()),   
]

