# Code Review

## CI test

- add 

## FastAPI Factory Pattern

- [official documentation](https://testdriven.io/courses/fastapi-celery/app-factory/#H-1-app-factory)
- Execute command:

```bash
uvicorn Fraud_Detection_BERT:create_app \
    --factory \
    --host 0.0.0.0 \
    --port 8000 \
    --reload    # under development env
```
