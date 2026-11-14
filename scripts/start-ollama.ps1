param(
    [ValidateSet('f16', 'q8_0', 'q4_0')]
    [string]$CacheType = 'q8_0'
)

# Run after quitting the Ollama tray app. This does not kill any process.
$ErrorActionPreference = 'Stop'
$ollamaCommand = Get-Command ollama -ErrorAction Stop
$env:OLLAMA_HOST = '127.0.0.1:11434'
$env:OLLAMA_FLASH_ATTENTION = '1'
$env:OLLAMA_KV_CACHE_TYPE = $CacheType
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_MAX_LOADED_MODELS = '1'
$env:OLLAMA_CONTEXT_LENGTH = '2048'
Write-Host "Requested server profile: flash=1, cache=$CacheType, parallel=1"
Write-Host 'Check Ollama logs for actual Flash Attention / KV cache support.'
& $ollamaCommand.Source serve
