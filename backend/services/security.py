from fastapi.security import OAuth2PasswordBearer
from passlib.context import CryptContext

pwd_hasher = CryptContext(schemes=["sha256_crypt", "md5_crypt", "des_crypt"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def crypt_password(plain_password: str):

    password_hashed = pwd_hasher.hash(plain_password, "sha256_crypt")
    return password_hashed


def verify_password(plain_password: str, hashed_pasword: str):
    return pwd_hasher.verify(plain_password, hashed_pasword)
