
from cryptography.fernet import Fernet

def generate_key():
    key = Fernet.generate_key()
    print("🔑 Key generated")
    return key


def encrypt_file(input_path, output_path, key):

    cipher = Fernet(key)

    with open(input_path, 'rb') as f:
        data = f.read()

    encrypted_data = cipher.encrypt(data)

    with open(output_path, 'wb') as f:
        f.write(encrypted_data)

    print(f"🔒 File encrypted → {output_path}")


def decrypt_file(input_path, output_path, key):

    cipher = Fernet(key)

    with open(input_path, 'rb') as f:
        data = f.read()

    decrypted_data = cipher.decrypt(data)

    with open(output_path, 'wb') as f:
        f.write(decrypted_data)

    print(f"🔓 File decrypted → {output_path}")
