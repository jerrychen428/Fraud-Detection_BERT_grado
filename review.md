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

## requirements.txt

- Current `requirements.txt` is not complete.
- `pip freeze > requirements.total.txt`
- Dependency management with more comprehensive control: [`poetry`](https://python-poetry.org/)

## pytest

- Utilize [`fixture`](https://docs.pytest.org/en/6.2.x/fixture.html) for reusable resources.
- [conftest.py](https://docs.pytest.org/en/stable/reference/fixtures.html): sharing fixtures across multiple files
