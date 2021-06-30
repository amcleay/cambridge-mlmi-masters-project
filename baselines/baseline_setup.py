"""This script should be run before dialogue generation scripts to circumvent convlab-2 bugs
where models are not automatically downloaded when models are initialised.
"""
from system_models import baseline_sys_model
from user_models import baseline_usr_model


def main():
    baseline_usr_model()
    baseline_sys_model()


if __name__ == "__main__":
    main()
