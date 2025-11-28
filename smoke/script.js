// 全局变量
let selectedCigarette = null;
let burnTimer = null;
let startTime = 0;
let totalDuration = 0;
let isBurning = false;

// DOM元素
const smokeButton = document.getElementById('smoke-button');
const resetButton = document.getElementById('reset-button');
const smokeAnimation = document.getElementById('smoke-animation');
const cigaretteItems = document.querySelectorAll('.cigarette-item');
const cigaretteBody = document.getElementById('cigarette-body');
const cigaretteBurn = document.getElementById('cigarette-burn');
const cigaretteFire = document.getElementById('cigarette-fire');
const cigaretteBrand = document.getElementById('cigarette-brand');
const timerDisplay = document.getElementById('timer-display');

// 为香烟品牌项目添加点击事件
cigaretteItems.forEach(item => {
    item.addEventListener('click', () => {
        // 如果正在燃烧，不允许更换香烟
        if (isBurning) return;
        
        // 移除所有选中状态
        cigaretteItems.forEach(i => i.classList.remove('selected'));
        // 添加当前选中状态
        item.classList.add('selected');
        // 保存选中的香烟信息
        selectedCigarette = {
            brand: item.dataset.brand,
            burnTime: parseInt(item.dataset.burnTime)
        };
        // 更新显示的品牌名称
        cigaretteBrand.textContent = selectedCigarette.brand;
        // 启用抽烟按钮
        smokeButton.disabled = false;
    });
});

// 为抽烟按钮添加点击事件
smokeButton.addEventListener('click', () => {
    if (!selectedCigarette || isBurning) return;
    
    // 开始燃烧
    startBurning();
});

// 为重置按钮添加点击事件
resetButton.addEventListener('click', () => {
    resetSimulation();
});

// 开始燃烧函数
function startBurning() {
    if (!selectedCigarette) return;
    
    isBurning = true;
    // 禁用相关按钮
    smokeButton.disabled = true;
    cigaretteItems.forEach(item => {
        item.style.pointerEvents = 'none';
        item.style.opacity = '0.5';
    });
    // 启用重置按钮
    resetButton.disabled = false;
    
    // 显示香烟
    cigaretteBody.classList.add('active');
    
    // 开始燃烧动画
    setTimeout(() => {
        cigaretteBurn.classList.add('burning');
        cigaretteFire.classList.add('burning');
        
        // 设置燃烧总时间（毫秒）
        totalDuration = selectedCigarette.burnTime * 1000;
        startTime = Date.now();
        
        // 更新燃烧动画持续时间
        const burnElement = document.querySelector('.cigarette-burn.burning');
        if (burnElement) {
            burnElement.style.animationDuration = `${totalDuration / 1000}s`;
        }
        
        // 开始计时器
        updateTimer();
        burnTimer = setInterval(updateTimer, 100);
        
        // 开始产生烟雾
        generateSmoke();
        
        // 燃烧结束
        setTimeout(() => {
            completeBurning();
        }, totalDuration);
    }, 500);
}

// 更新计时器
function updateTimer() {
    const elapsed = Date.now() - startTime;
    const remaining = Math.max(0, totalDuration - elapsed);
    const seconds = Math.ceil(remaining / 1000);
    
    // 更新显示
    timerDisplay.textContent = `剩余: ${seconds}秒`;
    
    // 更新进度条（如果需要）
    // 这里可以添加进度条逻辑
}

// 生成烟雾
function generateSmoke() {
    if (!isBurning) return;
    
    // 创建烟雾粒子
    const smokeParticle = document.createElement('div');
    smokeParticle.classList.add('smoke-particle');
    
    // 设置随机属性
    const size = Math.random() * 40 + 20; // 20px - 60px
    const variationX = Math.random() * 30 - 15; // -15px to 15px
    const duration = Math.random() * 2 + 2; // 2s - 4s
    const opacity = Math.random() * 0.4 + 0.3; // 0.3 - 0.7
    
    smokeParticle.style.width = `${size}px`;
    smokeParticle.style.height = `${size}px`;
    smokeParticle.style.transform = `translateX(${variationX}px)`;
    smokeParticle.style.animationDuration = `${duration}s`;
    smokeParticle.style.opacity = opacity;
    
    // 添加到动画区域
    smokeAnimation.appendChild(smokeParticle);
    
    // 动画结束后移除
    setTimeout(() => {
        if (smokeParticle.parentNode === smokeAnimation) {
            smokeAnimation.removeChild(smokeParticle);
        }
    }, duration * 1000);
    
    // 继续生成烟雾，直到燃烧结束
    if (isBurning) {
        setTimeout(generateSmoke, Math.random() * 500 + 300); // 300ms - 800ms
    }
}

// 完成燃烧
function completeBurning() {
    isBurning = false;
    
    // 停止计时器
    if (burnTimer) {
        clearInterval(burnTimer);
        burnTimer = null;
    }
    
    // 停止燃烧动画
    cigaretteBurn.classList.remove('burning');
    cigaretteFire.classList.remove('burning');
    
    // 更新计时器显示
    timerDisplay.textContent = '已燃尽';
    
    // 启用重置按钮
    resetButton.disabled = false;
}

// 重置模拟
function resetSimulation() {
    // 停止燃烧
    isBurning = false;
    
    // 停止计时器
    if (burnTimer) {
        clearInterval(burnTimer);
        burnTimer = null;
    }
    
    // 重置界面
    cigaretteBurn.classList.remove('burning');
    cigaretteFire.classList.remove('burning');
    cigaretteBody.classList.remove('active');
    timerDisplay.textContent = '剩余: 0秒';
    
    // 清空烟雾
    smokeAnimation.innerHTML = '';
    
    // 重置按钮状态
    smokeButton.disabled = !selectedCigarette;
    resetButton.disabled = true;
    cigaretteItems.forEach(item => {
        item.style.pointerEvents = 'auto';
        item.style.opacity = '1';
    });
}

// 添加页面加载时的初始化函数
window.addEventListener('load', () => {
    console.log('中国香烟模拟器已加载');
    // 初始化重置按钮为禁用状态
    resetButton.disabled = true;
});