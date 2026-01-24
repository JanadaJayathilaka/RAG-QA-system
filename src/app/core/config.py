def get_settings() -> Settings:
    """Get the application settings instance (singleton pattern).

    Returns:
        Settings instance with all configuration values loaded.
    """
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings