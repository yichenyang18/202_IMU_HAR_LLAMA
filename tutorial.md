### Termux set up
```
termux-setup-storage
```
```
pkg upgrade
```
```
pkg install git cmake golang
```

### Compile and install Ollama
```
git clone --depth 1 https://github.com/ollama/ollama.git
```
```
cd ollama
```
```
go generate ./..
```
```
go build .
```

### Run Ollama and LLM
```
./ollama serve &
```

```
./ollama run [your model] --verbose
```

### Customize or import model
.makefile
```
FROM [your model]
TEMPERATURE ...
SYSTEM ...
```

Termux
```
./ollama create [your model] -f [your .makefile]
```

