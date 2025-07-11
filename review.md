# Code Review

## CI test

- add cache argument in action

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

## OOP and state

- Too much unnecessary encapsulation in Instance, hard to manage and limited transparency.
-> Parameters OR instance attributes?
