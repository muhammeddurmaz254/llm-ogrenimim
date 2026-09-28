from langchain_core.rate_limiters import InMemoryRateLimiter

# NVIDIA ucretsiz katmani dakikada 40 istek, tum zincirler bu limiti paylasir
rate_limiter = InMemoryRateLimiter(requests_per_second=40 / 60)
