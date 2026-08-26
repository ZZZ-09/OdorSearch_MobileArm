@echo off
chcp 65001 >nul
echo ========================================
echo 3D Odor Source Search Demo
echo ========================================
python scripts/run_search.py --max-steps 3000 --seed 42 --save-history outputs/final_demo.npz --save-plot outputs/final_demo.png
if %errorlevel% == 0 (
    echo.
    echo 生成三维视图...
    python scripts/plot_3d.py --history outputs/final_demo.npz --output outputs/warehouse_3d.png
    echo 完成。结果保存在 outputs/ 目录。
) else (
    echo 仿真运行失败。
)
pause
