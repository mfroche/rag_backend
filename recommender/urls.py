from django.urls import path
from .views import DailyRecommendationsByDummyPatientView, DailyRecommenderByPatientView, PatientFoodIntakeRecommenderByDateView, DailyPatientFoodIntakeRecommenderView, WeeklyPatientFoodIntakeRecommenderView, MonthlyPatientFoodIntakeRecommenderView, GeneralPatientFoodIntakeRecommenderView, WeeklyRecommendationsByDummyPatientView

# BASE ENDPOINT: /api/recommend/

urlpatterns = [
    # Daily Total Intake & Food Recommendations By Patient 
    path("nutri-and-food/patient/<int:pk>/daily", DailyRecommenderByPatientView.as_view()),
    path("nutri-and-food/patient/<int:pk>/daily/", DailyRecommenderByPatientView.as_view()),

    # Daily Total Intake & Food Recommendations Using dummy data
    path("nutri-and-food/patient/dummy/daily", DailyRecommendationsByDummyPatientView.as_view()),
    path("nutri-and-food/patient/dummy/daily/", DailyRecommendationsByDummyPatientView.as_view()),

    # Weekly Total Intake & Food Recommendations Using dummy data
    path("nutri-and-food/patient/dummy/weekly", WeeklyRecommendationsByDummyPatientView.as_view()),
    path("nutri-and-food/patient/dummy/weekly/", WeeklyRecommendationsByDummyPatientView.as_view()),


    # [PREVIOUS]
    # Patient Daily food intake recommendations
    path("patient/<int:pk>/daily", DailyPatientFoodIntakeRecommenderView.as_view()),
    path("patient/<int:pk>/daily/", DailyPatientFoodIntakeRecommenderView.as_view()),

    # Patient food intake recommendations By Date
    path("patient/<int:pk>/daily/<str:date>", PatientFoodIntakeRecommenderByDateView.as_view()),
    path("patient/<int:pk>/daily/<str:date>/", PatientFoodIntakeRecommenderByDateView.as_view()),

    # Patient Weekly food intake recommendations
    path("patient/<int:pk>/weekly", WeeklyPatientFoodIntakeRecommenderView.as_view()),
    path("patient/<int:pk>/weekly/", WeeklyPatientFoodIntakeRecommenderView.as_view()),

    # Patient Monthly food intake recommendations
    path("patient/<int:pk>/monthly", MonthlyPatientFoodIntakeRecommenderView.as_view()),
    path("patient/<int:pk>/monthly/", MonthlyPatientFoodIntakeRecommenderView.as_view()),


    # General Patient recommendations for macros and micros
    path("patient/<int:pk>/general", GeneralPatientFoodIntakeRecommenderView.as_view()),
    path("patient/<int:pk>/general/", GeneralPatientFoodIntakeRecommenderView.as_view()),   


    
]

