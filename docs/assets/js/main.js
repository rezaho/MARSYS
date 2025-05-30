// Main JavaScript file for Multi-Agent Reasoning Systems (MARSYS) documentation

// Add any custom JavaScript functionality here
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded - MARSYS ready');

    // Smooth scrolling for internal links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add glassmorphism effect on scroll
    const header = document.querySelector('.md-header');
    if (header) {
        window.addEventListener('scroll', function() {
            if (window.scrollY > 50) {
                header.style.backdropFilter = 'blur(25px)';
                header.style.boxShadow = '0 8px 32px rgba(0, 0, 0, 0.1)';
            } else {
                header.style.backdropFilter = 'blur(20px)';
                header.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.1)';
            }
        });
    }

    // Enhance code blocks with copy functionality feedback
    const copyButtons = document.querySelectorAll('.md-clipboard');
    copyButtons.forEach(button => {
        button.addEventListener('click', function() {
            const icon = this.querySelector('.md-clipboard__message');
            if (icon) {
                icon.textContent = 'Copied!';
                setTimeout(() => {
                    icon.textContent = '';
                }, 2000);
            }
        });
    });
});

// Add loading animation for page transitions
document.addEventListener('turbo:load', function() {
    document.body.classList.add('loaded');
});

// Mermaid diagram initialization (if needed)
if (typeof mermaid !== 'undefined') {
    mermaid.initialize({ 
        startOnLoad: true,
        theme: 'default',
        themeVariables: {
            primaryColor: '#6366f1',
            primaryTextColor: '#1f2937',
            primaryBorderColor: '#e5e7eb',
            lineColor: '#9ca3af',
            secondaryColor: '#f3f4f6',
            tertiaryColor: '#fef3c7'
        }
    });
}

// Ensure navigation is visible on all pages including homepage
document.addEventListener('DOMContentLoaded', function() {
    // Force navigation visibility
    const primaryNav = document.querySelector('.md-nav--primary');
    const sidebar = document.querySelector('.md-sidebar--primary');
    
    if (primaryNav) {
        primaryNav.style.display = 'block';
        primaryNav.style.visibility = 'visible';
        primaryNav.style.opacity = '1';
    }
    
    if (sidebar) {
        sidebar.style.display = 'block';
        sidebar.style.visibility = 'visible';
    }
});




// Sidebar Toggle Fix
document.addEventListener('DOMContentLoaded', function() {
    // Ensure section toggles work independently from links
    const sectionItems = document.querySelectorAll('.md-nav__item--section');
    
    sectionItems.forEach(item => {
        const toggle = item.querySelector('.md-nav__toggle');
        const checkbox = toggle ? toggle.querySelector('input[type="checkbox"]') : null;
        const link = item.querySelector('.md-nav__link');
        
        if (toggle && checkbox && link) {
            // Prevent the link from being activated when clicking the toggle area
            toggle.addEventListener('click', function(e) {
                e.preventDefault();
                e.stopPropagation();
                checkbox.checked = !checkbox.checked;
                // Trigger change event for MkDocs Material theme
                checkbox.dispatchEvent(new Event('change', { bubbles: true }));
            });
            
            // Prevent navigation when clicking in the toggle area of the link
            link.addEventListener('click', function(e) {
                const rect = this.getBoundingClientRect();
                const clickX = e.clientX - rect.left;
                const toggleAreaWidth = 40; // Width of the toggle area
                
                if (clickX > rect.width - toggleAreaWidth) {
                    e.preventDefault();
                    e.stopPropagation();
                    checkbox.checked = !checkbox.checked;
                    checkbox.dispatchEvent(new Event('change', { bubbles: true }));
                }
            });
        }
    });
    
    // Fix for active section positioning
    const activeSection = document.querySelector('.md-nav__item--active.md-nav__item--section');
    if (activeSection) {
        const link = activeSection.querySelector('.md-nav__link');
        if (link) {
            // Ensure the text doesn't get centered
            link.style.textAlign = 'left';
            link.style.paddingLeft = '0.8rem';
        }
    }
});

// Add loading animation for page transitions
document.addEventListener('turbo:load', function() {
    document.body.classList.add('loaded');
});

// Mermaid diagram initialization (if needed)
if (typeof mermaid !== 'undefined') {
    mermaid.initialize({ 
        startOnLoad: true,
        theme: 'default',
        themeVariables: {
            primaryColor: '#6366f1',
            primaryTextColor: '#1f2937',
            primaryBorderColor: '#e5e7eb',
            lineColor: '#9ca3af',
            secondaryColor: '#f3f4f6',
            tertiaryColor: '#fef3c7'
        }
    });
}
