<!doctype html>
<html class="">
	<head>
		<meta charset="utf-8" />
		<meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no" />
		<meta name="description" content="We’re on a journey to advance and democratize artificial intelligence through open source and open science." />
		<meta property="fb:app_id" content="1321688464574422" />
		<meta name="twitter:card" content="summary_large_image" />
		<meta name="twitter:site" content="@huggingface" />
		<meta name="twitter:image" content="https://cdn-thumbnails.huggingface.co/social-thumbnails/datasets/BAAI/CCI3-HQ.png" />
		<meta property="og:title" content="lighteval_tasks_v2.py · BAAI/CCI3-HQ at main" />
		<meta property="og:type" content="website" />
		<meta property="og:url" content="https://huggingface.co/datasets/BAAI/CCI3-HQ/blob/main/lighteval_tasks_v2.py" />
		<meta property="og:image" content="https://cdn-thumbnails.huggingface.co/social-thumbnails/datasets/BAAI/CCI3-HQ.png" />

		<link rel="stylesheet" href="/front/build/kube-47f8e3c/style.css" />

		<link rel="preconnect" href="https://fonts.gstatic.com" />
		<link
			href="https://fonts.googleapis.com/css2?family=Source+Sans+Pro:ital,wght@0,200;0,300;0,400;0,600;0,700;0,900;1,200;1,300;1,400;1,600;1,700;1,900&display=swap"
			rel="stylesheet"
		/>
		<link
			href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap"
			rel="stylesheet"
		/>

		<link
			rel="preload"
			href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.12.0/katex.min.css"
			as="style"
			onload="this.onload=null;this.rel='stylesheet'"
		/>
		<noscript>
			<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.12.0/katex.min.css" />
		</noscript>

		<link rel="canonical" href="https://huggingface.co/datasets/BAAI/CCI3-HQ/blob/main/lighteval_tasks_v2.py">  <!-- HEAD_svelte-1oal594_START --><style>.blob-line-num::before {
			content: attr(data-line-num);
		}
	</style><!-- HEAD_svelte-1oal594_END -->

		<title>lighteval_tasks_v2.py · BAAI/CCI3-HQ at main</title>

		<script
			defer
			data-domain="huggingface.co"
			event-loggedIn="false"
			src="/js/script.pageview-props.js"
		></script>
		<script>
			window.plausible =
				window.plausible ||
				function () {
					(window.plausible.q = window.plausible.q || []).push(arguments);
				};
		</script>
		<script>
			window.hubConfig = JSON.parse(`{"features":{"signupDisabled":false},"sshGitUrl":"git@hf.co","moonHttpUrl":"https://huggingface.co","captchaApiKey":"bd5f2066-93dc-4bdd-a64b-a24646ca3859","captchaDisabledOnSignup":true,"datasetViewerPublicUrl":"https://datasets-server.huggingface.co","stripePublicKey":"pk_live_x2tdjFXBCvXo2FFmMybezpeM00J6gPCAAc","environment":"production","userAgent":"HuggingFace (production)","spacesIframeDomain":"hf.space","spacesApiUrl":"https://api.hf.space","docSearchKey":"ece5e02e57300e17d152c08056145326e90c4bff3dd07d7d1ae40cf1c8d39cb6","logoDev":{"apiUrl":"https://img.logo.dev/","apiKey":"pk_UHS2HZOeRnaSOdDp7jbd5w"}}`);
		</script>
		<script type="text/javascript" src="https://de5282c3ca0c.edge.sdk.awswaf.com/de5282c3ca0c/526cf06acb0d/challenge.js" defer></script>
	</head>
	<body class="flex flex-col min-h-dvh bg-white dark:bg-gray-950 text-black ViewerBlobPage">
		<div class="flex min-h-dvh flex-col">
	<div class="SVELTE_HYDRATER contents" data-target="MainHeader" data-props="{&quot;classNames&quot;:&quot;&quot;,&quot;isWide&quot;:false,&quot;isZh&quot;:false}"><header class="border-b border-gray-100 "><div class="w-full px-4 container flex h-16 items-center"><div class="flex flex-1 items-center"><a class="mr-5 flex flex-none items-center lg:mr-6" href="/"><img alt="Hugging Face's logo" class="w-7 md:mr-2" src="/front/assets/huggingface_logo-noborder.svg">
				<span class="hidden whitespace-nowrap text-lg font-bold md:block">Hugging Face</span></a>
			<div class="relative flex-1 lg:max-w-sm mr-2 sm:mr-4 md:mr-3 xl:mr-6"><input autocomplete="off" class="w-full dark:bg-gray-950 pl-8 form-input-alt h-9 pr-3 focus:shadow-xl " name="" placeholder="Search models, datasets, users..."   spellcheck="false" type="text" value="">
	<svg class="absolute left-2.5 text-gray-400 top-1/2 transform -translate-y-1/2" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path d="M30 28.59L22.45 21A11 11 0 1 0 21 22.45L28.59 30zM5 14a9 9 0 1 1 9 9a9 9 0 0 1-9-9z" fill="currentColor"></path></svg>
	</div>
			<div class="flex flex-none items-center justify-center p-0.5 place-self-stretch lg:hidden"><button class="relative z-40 flex h-6 w-8 items-center justify-center" type="button"><svg width="1em" height="1em" viewBox="0 0 10 10" class="text-xl" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" preserveAspectRatio="xMidYMid meet" fill="currentColor"><path fill-rule="evenodd" clip-rule="evenodd" d="M1.65039 2.9999C1.65039 2.8066 1.80709 2.6499 2.00039 2.6499H8.00039C8.19369 2.6499 8.35039 2.8066 8.35039 2.9999C8.35039 3.1932 8.19369 3.3499 8.00039 3.3499H2.00039C1.80709 3.3499 1.65039 3.1932 1.65039 2.9999ZM1.65039 4.9999C1.65039 4.8066 1.80709 4.6499 2.00039 4.6499H8.00039C8.19369 4.6499 8.35039 4.8066 8.35039 4.9999C8.35039 5.1932 8.19369 5.3499 8.00039 5.3499H2.00039C1.80709 5.3499 1.65039 5.1932 1.65039 4.9999ZM2.00039 6.6499C1.80709 6.6499 1.65039 6.8066 1.65039 6.9999C1.65039 7.1932 1.80709 7.3499 2.00039 7.3499H8.00039C8.19369 7.3499 8.35039 7.1932 8.35039 6.9999C8.35039 6.8066 8.19369 6.6499 8.00039 6.6499H2.00039Z"></path></svg>
		</button>

	</div></div>
		<nav aria-label="Main" class="ml-auto hidden lg:block"><ul class="flex items-center space-x-1.5 xl:space-x-2"><li><a class="group flex items-center px-2 py-0.5 dark:hover:text-gray-400 hover:text-indigo-700" href="/models"><svg class="mr-1.5 text-gray-400 group-hover:text-indigo-500" style="" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 24 24"><path class="uim-quaternary" d="M20.23 7.24L12 12L3.77 7.24a1.98 1.98 0 0 1 .7-.71L11 2.76c.62-.35 1.38-.35 2 0l6.53 3.77c.29.173.531.418.7.71z" opacity=".25" fill="currentColor"></path><path class="uim-tertiary" d="M12 12v9.5a2.09 2.09 0 0 1-.91-.21L4.5 17.48a2.003 2.003 0 0 1-1-1.73v-7.5a2.06 2.06 0 0 1 .27-1.01L12 12z" opacity=".5" fill="currentColor"></path><path class="uim-primary" d="M20.5 8.25v7.5a2.003 2.003 0 0 1-1 1.73l-6.62 3.82c-.275.13-.576.198-.88.2V12l8.23-4.76c.175.308.268.656.27 1.01z" fill="currentColor"></path></svg>
					Models</a>
			</li><li><a class="group flex items-center px-2 py-0.5 dark:hover:text-gray-400 hover:text-red-700" href="/datasets"><svg class="mr-1.5 text-gray-400 group-hover:text-red-500" style="" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 25 25"><ellipse cx="12.5" cy="5" fill="currentColor" fill-opacity="0.25" rx="7.5" ry="2"></ellipse><path d="M12.5 15C16.6421 15 20 14.1046 20 13V20C20 21.1046 16.6421 22 12.5 22C8.35786 22 5 21.1046 5 20V13C5 14.1046 8.35786 15 12.5 15Z" fill="currentColor" opacity="0.5"></path><path d="M12.5 7C16.6421 7 20 6.10457 20 5V11.5C20 12.6046 16.6421 13.5 12.5 13.5C8.35786 13.5 5 12.6046 5 11.5V5C5 6.10457 8.35786 7 12.5 7Z" fill="currentColor" opacity="0.5"></path><path d="M5.23628 12C5.08204 12.1598 5 12.8273 5 13C5 14.1046 8.35786 15 12.5 15C16.6421 15 20 14.1046 20 13C20 12.8273 19.918 12.1598 19.7637 12C18.9311 12.8626 15.9947 13.5 12.5 13.5C9.0053 13.5 6.06886 12.8626 5.23628 12Z" fill="currentColor"></path></svg>
					Datasets</a>
			</li><li><a class="group flex items-center px-2 py-0.5 dark:hover:text-gray-400 hover:text-blue-700" href="/spaces"><svg class="mr-1.5 text-gray-400 group-hover:text-blue-500" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" viewBox="0 0 25 25"><path opacity=".5" d="M6.016 14.674v4.31h4.31v-4.31h-4.31ZM14.674 14.674v4.31h4.31v-4.31h-4.31ZM6.016 6.016v4.31h4.31v-4.31h-4.31Z" fill="currentColor"></path><path opacity=".75" fill-rule="evenodd" clip-rule="evenodd" d="M3 4.914C3 3.857 3.857 3 4.914 3h6.514c.884 0 1.628.6 1.848 1.414a5.171 5.171 0 0 1 7.31 7.31c.815.22 1.414.964 1.414 1.848v6.514A1.914 1.914 0 0 1 20.086 22H4.914A1.914 1.914 0 0 1 3 20.086V4.914Zm3.016 1.102v4.31h4.31v-4.31h-4.31Zm0 12.968v-4.31h4.31v4.31h-4.31Zm8.658 0v-4.31h4.31v4.31h-4.31Zm0-10.813a2.155 2.155 0 1 1 4.31 0 2.155 2.155 0 0 1-4.31 0Z" fill="currentColor"></path><path opacity=".25" d="M16.829 6.016a2.155 2.155 0 1 0 0 4.31 2.155 2.155 0 0 0 0-4.31Z" fill="currentColor"></path></svg>
					Spaces</a>
			</li><li><a class="group flex items-center px-2 py-0.5 dark:hover:text-gray-400 hover:text-yellow-700" href="/posts"><svg class="mr-1.5 text-gray-400 group-hover:text-yellow-500 !text-yellow-500" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" viewBox="0 0 12 12" preserveAspectRatio="xMidYMid meet"><path fill="currentColor" fill-rule="evenodd" d="M3.73 2.4A4.25 4.25 0 1 1 6 10.26H2.17l-.13-.02a.43.43 0 0 1-.3-.43l.01-.06a.43.43 0 0 1 .12-.22l.84-.84A4.26 4.26 0 0 1 3.73 2.4Z" clip-rule="evenodd"></path></svg>
					Posts</a>
			</li><li><a class="group flex items-center px-2 py-0.5 dark:hover:text-gray-400 hover:text-yellow-700" href="/docs"><svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" class="mr-1.5 text-gray-400 group-hover:text-yellow-500" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path opacity="0.5" d="M20.9022 5.10334L10.8012 10.8791L7.76318 9.11193C8.07741 8.56791 8.5256 8.11332 9.06512 7.7914L15.9336 3.73907C17.0868 3.08811 18.5002 3.26422 19.6534 3.91519L19.3859 3.73911C19.9253 4.06087 20.5879 4.56025 20.9022 5.10334Z" fill="currentColor"></path><path d="M10.7999 10.8792V28.5483C10.2136 28.5475 9.63494 28.4139 9.10745 28.1578C8.5429 27.8312 8.074 27.3621 7.74761 26.7975C7.42122 26.2327 7.24878 25.5923 7.24756 24.9402V10.9908C7.25062 10.3319 7.42358 9.68487 7.74973 9.1123L10.7999 10.8792Z" fill="currentColor" fill-opacity="0.75"></path><path fill-rule="evenodd" clip-rule="evenodd" d="M21.3368 10.8499V6.918C21.3331 6.25959 21.16 5.61234 20.8346 5.03949L10.7971 10.8727L10.8046 10.874L21.3368 10.8499Z" fill="currentColor"></path><path opacity="0.5" d="M21.7937 10.8488L10.7825 10.8741V28.5486L21.7937 28.5234C23.3344 28.5234 24.5835 27.2743 24.5835 25.7335V13.6387C24.5835 12.0979 23.4365 11.1233 21.7937 10.8488Z" fill="currentColor"></path></svg>
					Docs</a>
			</li>
		<li class="max-2xl:hidden"><div class="relative ">
	<button class="px-2 py-0.5 group hover:text-green-700 dark:hover:text-gray-400 flex items-center " type="button">
		<svg class="mr-1.5 text-gray-400 group-hover:text-green-500" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 24 24"><path class="uim-tertiary" d="M19 6H5a3 3 0 0 0-3 3v2.72L8.837 14h6.326L22 11.72V9a3 3 0 0 0-3-3z" opacity=".5" fill="currentColor"></path><path class="uim-primary" d="M10 6V5h4v1h2V5a2.002 2.002 0 0 0-2-2h-4a2.002 2.002 0 0 0-2 2v1h2zm-1.163 8L2 11.72V18a3.003 3.003 0 0 0 3 3h14a3.003 3.003 0 0 0 3-3v-6.28L15.163 14H8.837z" fill="currentColor"></path></svg>
			Solutions
		</button>
	
	
	</div></li>
		<li><a class="group flex items-center px-2 py-0.5 hover:text-gray-500 dark:hover:text-gray-400" href="/pricing">Pricing
			</a></li>

		<li><div class="relative group">
	<button class="px-2 py-0.5 hover:text-gray-500 dark:hover:text-gray-600 flex items-center " type="button">
		<svg class=" text-gray-500 w-5 group-hover:text-gray-400 dark:text-gray-300 dark:group-hover:text-gray-400" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" viewBox="0 0 32 18" preserveAspectRatio="xMidYMid meet"><path fill-rule="evenodd" clip-rule="evenodd" d="M14.4504 3.30221C14.4504 2.836 14.8284 2.45807 15.2946 2.45807H28.4933C28.9595 2.45807 29.3374 2.836 29.3374 3.30221C29.3374 3.76842 28.9595 4.14635 28.4933 4.14635H15.2946C14.8284 4.14635 14.4504 3.76842 14.4504 3.30221Z" fill="currentColor"></path><path fill-rule="evenodd" clip-rule="evenodd" d="M14.4504 9.00002C14.4504 8.53382 14.8284 8.15588 15.2946 8.15588H28.4933C28.9595 8.15588 29.3374 8.53382 29.3374 9.00002C29.3374 9.46623 28.9595 9.84417 28.4933 9.84417H15.2946C14.8284 9.84417 14.4504 9.46623 14.4504 9.00002Z" fill="currentColor"></path><path fill-rule="evenodd" clip-rule="evenodd" d="M14.4504 14.6978C14.4504 14.2316 14.8284 13.8537 15.2946 13.8537H28.4933C28.9595 13.8537 29.3374 14.2316 29.3374 14.6978C29.3374 15.164 28.9595 15.542 28.4933 15.542H15.2946C14.8284 15.542 14.4504 15.164 14.4504 14.6978Z" fill="currentColor"></path><path fill-rule="evenodd" clip-rule="evenodd" d="M1.94549 6.87377C2.27514 6.54411 2.80962 6.54411 3.13928 6.87377L6.23458 9.96907L9.32988 6.87377C9.65954 6.54411 10.194 6.54411 10.5237 6.87377C10.8533 7.20343 10.8533 7.73791 10.5237 8.06756L6.23458 12.3567L1.94549 8.06756C1.61583 7.73791 1.61583 7.20343 1.94549 6.87377Z" fill="currentColor"></path></svg>
			
		</button>
	
	
	</div></li>
		<li><hr class="h-5 w-0.5 border-none bg-gray-100 dark:bg-gray-800"></li>
		<li><a class="block cursor-pointer px-2 py-0.5 hover:text-gray-500 dark:hover:text-gray-400" href="/login">Log In
				</a></li>
			<li><a class="rounded-full border border-transparent bg-gray-900 px-3 py-1 leading-none text-white hover:border-black hover:bg-white hover:text-black" href="/join">Sign Up
					</a></li></ul></nav></div></header></div>
	
	
	
	<div class="SVELTE_HYDRATER contents" data-target="SSOBanner" data-props="{}"></div>
	

	<main class="flex flex-1 flex-col"><div class="SVELTE_HYDRATER contents" data-target="DatasetHeader" data-props="{&quot;activeTab&quot;:&quot;files&quot;,&quot;author&quot;:{&quot;avatarUrl&quot;:&quot;https://cdn-avatars.huggingface.co/v1/production/uploads/1664511063789-632c234f42c386ebd2710434.png&quot;,&quot;fullname&quot;:&quot;Beijing Academy of Artificial Intelligence&quot;,&quot;name&quot;:&quot;BAAI&quot;,&quot;type&quot;:&quot;org&quot;,&quot;isHf&quot;:false,&quot;isMod&quot;:false,&quot;isEnterprise&quot;:false},&quot;canReadRepoSettings&quot;:false,&quot;dataset&quot;:{&quot;author&quot;:&quot;BAAI&quot;,&quot;cardData&quot;:{&quot;task_categories&quot;:[&quot;text-generation&quot;],&quot;language&quot;:[&quot;zh&quot;],&quot;dataset_info&quot;:{&quot;features&quot;:[{&quot;name&quot;:&quot;id&quot;,&quot;dtype&quot;:&quot;string&quot;},{&quot;name&quot;:&quot;text&quot;,&quot;dtype&quot;:&quot;string&quot;},{&quot;name&quot;:&quot;score&quot;,&quot;dtype&quot;:&quot;float&quot;}],&quot;splits&quot;:[{&quot;name&quot;:&quot;train&quot;}]},&quot;configs&quot;:[{&quot;config_name&quot;:&quot;default&quot;,&quot;data_files&quot;:[{&quot;split&quot;:&quot;train&quot;,&quot;path&quot;:&quot;data/part_*&quot;}]}]},&quot;cardExists&quot;:true,&quot;createdAt&quot;:&quot;2024-09-19T05:33:35.000Z&quot;,&quot;description&quot;:&quot;\n\t\n\t\t\n\t\n\t\n\t\tData Description\n\t\n\nTo address the scarcity of high-quality safety datasets in the Chinese, we open-sourced the CCI (Chinese Corpora Internet) dataset on November 29, 2023. \nBuilding on this foundation, we continue to expand the data source, adopt stricter data cleaning methods, and complete the construction of the CCI 3.0 dataset. This dataset is composed of high-quality, reliable Internet data from trusted sources. \nAnd then with more stricter filtering, The CCI 3.0 HQ corpus… See the full description on the dataset page: https://huggingface.co/datasets/BAAI/CCI3-HQ.&quot;,&quot;downloads&quot;:131,&quot;downloadsAllTime&quot;:131,&quot;id&quot;:&quot;BAAI/CCI3-HQ&quot;,&quot;isLikedByUser&quot;:false,&quot;isWatchedByUser&quot;:false,&quot;lastModified&quot;:&quot;2024-09-26T08:24:16.000Z&quot;,&quot;likes&quot;:5,&quot;datasetsServerInfo&quot;:{&quot;viewer&quot;:&quot;viewer-partial&quot;,&quot;numRows&quot;:54791115,&quot;libraries&quot;:[&quot;datasets&quot;,&quot;dask&quot;,&quot;mlcroissant&quot;],&quot;formats&quot;:[&quot;json&quot;],&quot;modalities&quot;:[&quot;text&quot;]},&quot;discussionsDisabled&quot;:false,&quot;repoType&quot;:&quot;dataset&quot;,&quot;private&quot;:false,&quot;gated&quot;:false,&quot;tags&quot;:[&quot;task_categories:text-generation&quot;,&quot;language:zh&quot;,&quot;size_categories:10M<n<100M&quot;,&quot;format:json&quot;,&quot;modality:text&quot;,&quot;library:datasets&quot;,&quot;library:dask&quot;,&quot;library:mlcroissant&quot;,&quot;region:us&quot;],&quot;tag_objs&quot;:[{&quot;id&quot;:&quot;task_categories:text-generation&quot;,&quot;label&quot;:&quot;text-generation&quot;,&quot;type&quot;:&quot;task_categories&quot;,&quot;subType&quot;:&quot;nlp&quot;},{&quot;id&quot;:&quot;language:zh&quot;,&quot;label&quot;:&quot;Chinese&quot;,&quot;type&quot;:&quot;language&quot;},{&quot;id&quot;:&quot;size_categories:10M<n<100M&quot;,&quot;label&quot;:&quot;10M - 100M&quot;,&quot;type&quot;:&quot;size_categories&quot;},{&quot;id&quot;:&quot;format:json&quot;,&quot;label&quot;:&quot;json&quot;,&quot;type&quot;:&quot;format&quot;},{&quot;id&quot;:&quot;modality:text&quot;,&quot;label&quot;:&quot;Text&quot;,&quot;type&quot;:&quot;modality&quot;},{&quot;id&quot;:&quot;library:datasets&quot;,&quot;label&quot;:&quot;Datasets&quot;,&quot;type&quot;:&quot;library&quot;},{&quot;id&quot;:&quot;library:dask&quot;,&quot;label&quot;:&quot;Dask&quot;,&quot;type&quot;:&quot;library&quot;},{&quot;id&quot;:&quot;library:mlcroissant&quot;,&quot;label&quot;:&quot;Croissant&quot;,&quot;type&quot;:&quot;library&quot;},{&quot;type&quot;:&quot;region&quot;,&quot;label&quot;:&quot;🇺🇸 Region: US&quot;,&quot;id&quot;:&quot;region:us&quot;}],&quot;hasBlockedOids&quot;:false},&quot;discussionsStats&quot;:{&quot;closed&quot;:0,&quot;open&quot;:1,&quot;total&quot;:1}}"><header class="from-gray-50-to-white border-b border-gray-100 bg-gradient-to-t via-white dark:via-gray-950 pt-6 sm:pt-9"><div class="container relative "><h1 class="flex flex-wrap items-center leading-tight mb-3 text-lg max-sm:gap-y-1.5 md:text-xl"><a href="/datasets" class="group flex items-center"><svg class="mr-1.5 text-gray-400" style="" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 25 25"><ellipse cx="12.5" cy="5" fill="currentColor" fill-opacity="0.25" rx="7.5" ry="2"></ellipse><path d="M12.5 15C16.6421 15 20 14.1046 20 13V20C20 21.1046 16.6421 22 12.5 22C8.35786 22 5 21.1046 5 20V13C5 14.1046 8.35786 15 12.5 15Z" fill="currentColor" opacity="0.5"></path><path d="M12.5 7C16.6421 7 20 6.10457 20 5V11.5C20 12.6046 16.6421 13.5 12.5 13.5C8.35786 13.5 5 12.6046 5 11.5V5C5 6.10457 8.35786 7 12.5 7Z" fill="currentColor" opacity="0.5"></path><path d="M5.23628 12C5.08204 12.1598 5 12.8273 5 13C5 14.1046 8.35786 15 12.5 15C16.6421 15 20 14.1046 20 13C20 12.8273 19.918 12.1598 19.7637 12C18.9311 12.8626 15.9947 13.5 12.5 13.5C9.0053 13.5 6.06886 12.8626 5.23628 12Z" fill="currentColor"></path></svg>
					<span class="mr-2.5 font-semibold text-gray-400 group-hover:text-gray-500">Datasets:</span></a>
			<div class="group flex flex-none items-center"><div class="relative mr-1.5 flex items-center">

			<img alt="" class="w-3.5 h-3.5 rounded  flex-none" src="https://cdn-avatars.huggingface.co/v1/production/uploads/1664511063789-632c234f42c386ebd2710434.png" crossorigin="anonymous"></div>
		<a href="/BAAI" class="text-gray-400 hover:text-blue-600">BAAI</a>
		<div class="mx-0.5 text-gray-300">/</div></div>

<div class="max-w-full "><a class="break-words font-mono font-semibold hover:text-blue-600 " href="/datasets/BAAI/CCI3-HQ">CCI3-HQ</a>
	<button class="relative text-sm mr-4 inline-flex cursor-pointer items-center text-sm focus:outline-none  mx-0.5   text-gray-600 " title="Copy dataset name to clipboard" type="button"><svg class="" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" fill="currentColor" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path d="M28,10V28H10V10H28m0-2H10a2,2,0,0,0-2,2V28a2,2,0,0,0,2,2H28a2,2,0,0,0,2-2V10a2,2,0,0,0-2-2Z" transform="translate(0)"></path><path d="M4,18H2V4A2,2,0,0,1,4,2H18V4H4Z" transform="translate(0)"></path><rect fill="none" width="32" height="32"></rect></svg>
	
	</button></div>
			<div class="inline-flex items-center overflow-hidden whitespace-nowrap rounded-md border bg-white text-sm leading-none text-gray-500  mr-2"><button class="relative flex items-center overflow-hidden from-red-50 to-transparent dark:from-red-900 px-1.5 py-1 hover:bg-gradient-to-t focus:outline-none"  title="Like"><svg class="left-1.5 absolute" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32" fill="currentColor"><path d="M22.45,6a5.47,5.47,0,0,1,3.91,1.64,5.7,5.7,0,0,1,0,8L16,26.13,5.64,15.64a5.7,5.7,0,0,1,0-8,5.48,5.48,0,0,1,7.82,0L16,10.24l2.53-2.58A5.44,5.44,0,0,1,22.45,6m0-2a7.47,7.47,0,0,0-5.34,2.24L16,7.36,14.89,6.24a7.49,7.49,0,0,0-10.68,0,7.72,7.72,0,0,0,0,10.82L16,29,27.79,17.06a7.72,7.72,0,0,0,0-10.82A7.49,7.49,0,0,0,22.45,4Z"></path></svg>

		
		<span class="ml-4 pl-0.5 ">like</span></button>
	<button class="flex items-center border-l px-1.5 py-1 text-gray-400 hover:bg-gray-50 focus:bg-gray-100 focus:outline-none dark:hover:bg-gray-900 dark:focus:bg-gray-800" title="See users who liked this repository">5</button></div>




			
	</h1>
		<div class="mb-3 flex flex-wrap md:mb-4"><div class="mr-1 flex flex-wrap items-center"><span class="mb-1 mr-1 p-1 text-sm leading-tight text-gray-400 md:mb-1.5">Tasks:
	</span>
	<a class="mb-1 mr-1 md:mb-1.5 md:mr-1.5 rounded-lg" href="/datasets?task_categories=task_categories%3Atext-generation"><div class="tag tag-white   "><div class="tag-ico -ml-2 tag-ico-indigo"><svg class="" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" fill="currentColor" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 18 18"><path d="M16.2607 8.08202L14.468 6.28928C14.3063 6.12804 14.0873 6.03749 13.859 6.03749C13.6307 6.03749 13.4117 6.12804 13.25 6.28928L5.6375 13.904V16.9125H8.64607L16.2607 9.30002C16.422 9.13836 16.5125 8.91935 16.5125 8.69102C16.5125 8.4627 16.422 8.24369 16.2607 8.08202V8.08202ZM8.1953 15.825H6.725V14.3547L11.858 9.22118L13.3288 10.6915L8.1953 15.825ZM14.0982 9.92262L12.6279 8.45232L13.8606 7.21964L15.3309 8.68994L14.0982 9.92262Z"></path><path d="M6.18125 9.84373H7.26875V6.03748H8.9V4.94998H4.55V6.03748H6.18125V9.84373Z"></path><path d="M4.55 11.475H2.375V2.775H11.075V4.95H12.1625V2.775C12.1625 2.48658 12.0479 2.20997 11.844 2.00602C11.64 1.80208 11.3634 1.6875 11.075 1.6875H2.375C2.08658 1.6875 1.80997 1.80208 1.60602 2.00602C1.40207 2.20997 1.2875 2.48658 1.2875 2.775V11.475C1.2875 11.7634 1.40207 12.04 1.60602 12.244C1.80997 12.4479 2.08658 12.5625 2.375 12.5625H4.55V11.475Z"></path></svg></div>

	

	<span>Text Generation</span>
	

	</div></a>

	</div><div class="mr-1 flex flex-wrap items-center"><span class="mb-1 mr-1 p-1 text-sm leading-tight text-gray-400 md:mb-1.5">Modalities:
	</span>
	<a class="mb-1 mr-1 md:mb-1.5 md:mr-1.5 rounded-lg" href="/datasets?modality=modality%3Atext"><div class="tag tag-white   ">
		<svg class="text-red-700 dark:text-red-600" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path fill-rule="evenodd" clip-rule="evenodd" d="M4.619 4.619C2.667 6.573 2.667 9.715 2.667 16s0 9.428 1.952 11.38C6.573 29.333 9.715 29.333 16 29.333s9.428 0 11.38-1.953c1.953-1.95 1.953-5.095 1.953-11.38s0-9.428-1.953-11.381C25.43 2.667 22.285 2.667 16 2.667s-9.428 0-11.381 1.952m8.65 3.714c-.573 0-1.109 0-1.546.066-.495.073-1.003.248-1.41.7-.392.436-.53.956-.59 1.452-.056.464-.056 1.04-.056 1.689V13a1 1 0 1 0 2 0v-.704c0-.724.001-1.176.041-1.505q.015-.15.061-.294a.2.2 0 0 1 .031-.061q0-.003.016-.01a.8.8 0 0 1 .203-.05c.272-.04.654-.043 1.314-.043H15v11.334h-2.333a1 1 0 1 0 0 2H20a1 1 0 0 0 0-2h-3V10.333h1.667c.66 0 1.042.003 1.314.043.123.019.18.04.203.05l.015.009a.2.2 0 0 1 .032.061c.018.05.042.14.061.295.04.329.041.781.041 1.506V13a1 1 0 1 0 2 0v-.76c0-.65 0-1.225-.056-1.69-.06-.495-.198-1.015-.59-1.453-.407-.45-.915-.625-1.41-.698-.437-.067-.973-.067-1.546-.066z" fill="currentColor"></path></svg>

	

	<span>Text</span>
	

	</div></a>

	</div><div class="mr-1 flex flex-wrap items-center"><span class="mb-1 mr-1 p-1 text-sm leading-tight text-gray-400 md:mb-1.5">Formats:
	</span>
	<a class="mb-1 mr-1 md:mb-1.5 md:mr-1.5 rounded-lg" href="/datasets?format=format%3Ajson"><div class="tag tag-white   ">
		<svg class="" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 12 12"><path fill-rule="evenodd" clip-rule="evenodd" d="M8.917 2.25h-.834v.833h.834v2.084A.833.833 0 0 0 9.75 6a.833.833 0 0 0-.833.833v2.084h-.834v.833h.834c.446-.113.833-.375.833-.833V7.25a.833.833 0 0 1 .833-.833H11v-.834h-.417a.833.833 0 0 1-.833-.833V3.083a.833.833 0 0 0-.833-.833Zm-5.834 0a.833.833 0 0 0-.833.833V4.75a.833.833 0 0 1-.833.833H1v.834h.417a.833.833 0 0 1 .833.833v1.667a.833.833 0 0 0 .833.833h.834v-.833h-.834V6.833A.833.833 0 0 0 2.25 6a.833.833 0 0 0 .833-.833V3.083h.834V2.25h-.834ZM6 7.25a.417.417 0 1 0 0 .833.417.417 0 0 0 0-.833Zm1.667 0a.417.417 0 1 0 0 .833.417.417 0 0 0 0-.833Zm-3.334 0a.417.417 0 1 0 0 .833.417.417 0 0 0 0-.833Z" fill="currentColor"></path></svg>

	

	<span>json</span>
	

	</div></a>

	</div><div class="mr-1 flex flex-wrap items-center"><span class="mb-1 mr-1 p-1 text-sm leading-tight text-gray-400 md:mb-1.5">Languages:
	</span>
	<a class="mb-1 mr-1 md:mb-1.5 md:mr-1.5 rounded-lg" href="/datasets?language=language%3Azh"><div class="tag tag-white   ">
		<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" class="text-green-600/80" preserveAspectRatio="xMidYMid meet" width="1em" height="1em" viewBox="0 0 10 10"><path fill-rule="evenodd" clip-rule="evenodd" d="M0.625 5C0.625 6.16032 1.08594 7.27312 1.90641 8.09359C2.72688 8.91406 3.83968 9.375 5 9.375C6.16032 9.375 7.27312 8.91406 8.09359 8.09359C8.91406 7.27312 9.375 6.16032 9.375 5C9.375 3.83968 8.91406 2.72688 8.09359 1.90641C7.27312 1.08594 6.16032 0.625 5 0.625C3.83968 0.625 2.72688 1.08594 1.90641 1.90641C1.08594 2.72688 0.625 3.83968 0.625 5ZM7.64365 7.48027C7.61734 7.50832 7.59054 7.53598 7.56326 7.56326C7.13828 7.98824 6.61864 8.2968 6.0539 8.46842C6.29802 8.11949 6.49498 7.64804 6.63475 7.09483C7.00845 7.18834 7.35014 7.3187 7.64365 7.48027ZM8.10076 6.87776C8.37677 6.42196 8.55005 5.90894 8.60556 5.37499H6.86808C6.85542 5.71597 6.82551 6.04557 6.77971 6.35841C7.25309 6.47355 7.68808 6.6414 8.062 6.85549C8.07497 6.86283 8.08789 6.87025 8.10076 6.87776ZM6.03795 6.22536C6.07708 5.95737 6.1044 5.67232 6.11705 5.37499H3.88295C3.89666 5.69742 3.92764 6.00542 3.9722 6.29287C4.37075 6.21726 4.79213 6.17749 5.224 6.17749C5.50054 6.17749 5.77294 6.19376 6.03795 6.22536ZM4.1261 7.02673C4.34894 7.84835 4.68681 8.375 5 8.375C5.32122 8.375 5.66839 7.82101 5.8908 6.963C5.67389 6.93928 5.45082 6.92699 5.224 6.92699C4.84316 6.92699 4.47332 6.96176 4.1261 7.02673ZM3.39783 7.21853C3.53498 7.71842 3.72038 8.14579 3.9461 8.46842C3.42141 8.30898 2.93566 8.03132 2.52857 7.65192C2.77253 7.48017 3.06711 7.33382 3.39783 7.21853ZM3.23916 6.48077C3.18263 6.13193 3.14625 5.76074 3.13192 5.37499H1.39444C1.4585 5.99112 1.67936 6.57938 2.03393 7.08403C2.3706 6.83531 2.78055 6.63162 3.23916 6.48077ZM1.39444 4.62499H3.13192C3.14615 4.24204 3.18211 3.87344 3.23794 3.52681C2.77814 3.37545 2.36731 3.17096 2.03024 2.92123C1.67783 3.42469 1.45828 4.011 1.39444 4.62499ZM2.5237 2.35262C2.76812 2.52552 3.06373 2.67281 3.39584 2.78875C3.53318 2.28573 3.71928 1.85578 3.9461 1.53158C3.41932 1.69166 2.93178 1.97089 2.5237 2.35262ZM3.97101 3.71489C3.92709 4.00012 3.89654 4.30547 3.88295 4.62499H6.11705C6.10453 4.33057 6.07761 4.04818 6.03909 3.78248C5.77372 3.81417 5.50093 3.83049 5.224 3.83049C4.79169 3.83049 4.3699 3.79065 3.97101 3.71489ZM5.8928 3.04476C5.67527 3.06863 5.45151 3.08099 5.224 3.08099C4.84241 3.08099 4.47186 3.04609 4.12405 2.98086C4.34686 2.1549 4.68584 1.625 5 1.625C5.32218 1.625 5.67048 2.18233 5.8928 3.04476ZM6.78083 3.6493C6.826 3.95984 6.85552 4.28682 6.86808 4.62499H8.60556C8.55029 4.09337 8.37827 3.58251 8.10436 3.1282C8.0903 3.1364 8.07618 3.14449 8.062 3.15249C7.68838 3.36641 7.25378 3.53417 6.78083 3.6493ZM7.64858 2.52499C7.35446 2.68754 7.0117 2.81868 6.63664 2.91268C6.49676 2.35623 6.29913 1.88209 6.0539 1.53158C6.61864 1.7032 7.13828 2.01176 7.56326 2.43674C7.59224 2.46572 7.62068 2.49514 7.64858 2.52499Z" fill="currentColor"></path></svg>

	

	<span>Chinese</span>
	

	</div></a>

	</div><div class="mr-1 flex flex-wrap items-center"><span class="mb-1 mr-1 p-1 text-sm leading-tight text-gray-400 md:mb-1.5">Size:
	</span>
	<a class="mb-1 mr-1 md:mb-1.5 md:mr-1.5 rounded-lg" href="/datasets?size_categories=size_categories%3A10M%3Cn%3C100M"><div class="tag tag-white   ">

	

	<span>10M - 100M</span>
	

	</div></a>

	</div><div class="mr-1 flex flex-wrap items-center"><span class="mb-1 mr-1 p-1 text-sm leading-tight text-gray-400 md:mb-1.5">Libraries:
	</span>
	<a class="mb-1 mr-1 md:mb-1.5 md:mr-1.5 rounded-lg" href="/datasets?library=library%3Adatasets"><div class="tag tag-white   "><svg class="text-black inline-block text-sm" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" preserveAspectRatio="xMidYMid meet" width="1em" height="1em" viewBox="0 0 95 88"><path fill="#fff" d="M94.25 70.08a8.28 8.28 0 0 1-.43 6.46 10.57 10.57 0 0 1-3 3.6 25.18 25.18 0 0 1-5.7 3.2 65.74 65.74 0 0 1-7.56 2.65 46.67 46.67 0 0 1-11.42 1.68c-5.42.05-10.09-1.23-13.4-4.5a40.4 40.4 0 0 1-10.14.03c-3.34 3.25-7.99 4.52-13.39 4.47a46.82 46.82 0 0 1-11.43-1.68 66.37 66.37 0 0 1-7.55-2.65c-2.28-.98-4.17-2-5.68-3.2a10.5 10.5 0 0 1-3.02-3.6c-.99-2-1.18-4.3-.42-6.46a8.54 8.54 0 0 1-.33-5.63c.25-.95.66-1.83 1.18-2.61a8.67 8.67 0 0 1 2.1-8.47 8.23 8.23 0 0 1 2.82-2.07 41.75 41.75 0 1 1 81.3-.12 8.27 8.27 0 0 1 3.11 2.19 8.7 8.7 0 0 1 2.1 8.47c.52.78.93 1.66 1.18 2.61a8.61 8.61 0 0 1-.32 5.63Z"></path><path fill="#FFD21E" d="M47.21 76.5a34.75 34.75 0 1 0 0-69.5 34.75 34.75 0 0 0 0 69.5Z"></path><path fill="#FF9D0B" d="M81.96 41.75a34.75 34.75 0 1 0-69.5 0 34.75 34.75 0 0 0 69.5 0Zm-73.5 0a38.75 38.75 0 1 1 77.5 0 38.75 38.75 0 0 1-77.5 0Z"></path><path fill="#3A3B45" d="M58.5 32.3c1.28.44 1.78 3.06 3.07 2.38a5 5 0 1 0-6.76-2.07c.61 1.15 2.55-.72 3.7-.32ZM34.95 32.3c-1.28.44-1.79 3.06-3.07 2.38a5 5 0 1 1 6.76-2.07c-.61 1.15-2.56-.72-3.7-.32Z"></path><path fill="#FF323D" d="M46.96 56.29c9.83 0 13-8.76 13-13.26 0-2.34-1.57-1.6-4.09-.36-2.33 1.15-5.46 2.74-8.9 2.74-7.19 0-13-6.88-13-2.38s3.16 13.26 13 13.26Z"></path><path fill="#3A3B45" fill-rule="evenodd" d="M39.43 54a8.7 8.7 0 0 1 5.3-4.49c.4-.12.81.57 1.24 1.28.4.68.82 1.37 1.24 1.37.45 0 .9-.68 1.33-1.35.45-.7.89-1.38 1.32-1.25a8.61 8.61 0 0 1 5 4.17c3.73-2.94 5.1-7.74 5.1-10.7 0-2.34-1.57-1.6-4.09-.36l-.14.07c-2.31 1.15-5.39 2.67-8.77 2.67s-6.45-1.52-8.77-2.67c-2.6-1.29-4.23-2.1-4.23.29 0 3.05 1.46 8.06 5.47 10.97Z" clip-rule="evenodd"></path><path fill="#FF9D0B" d="M70.71 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5ZM24.21 37a3.25 3.25 0 1 0 0-6.5 3.25 3.25 0 0 0 0 6.5ZM17.52 48c-1.62 0-3.06.66-4.07 1.87a5.97 5.97 0 0 0-1.33 3.76 7.1 7.1 0 0 0-1.94-.3c-1.55 0-2.95.59-3.94 1.66a5.8 5.8 0 0 0-.8 7 5.3 5.3 0 0 0-1.79 2.82c-.24.9-.48 2.8.8 4.74a5.22 5.22 0 0 0-.37 5.02c1.02 2.32 3.57 4.14 8.52 6.1 3.07 1.22 5.89 2 5.91 2.01a44.33 44.33 0 0 0 10.93 1.6c5.86 0 10.05-1.8 12.46-5.34 3.88-5.69 3.33-10.9-1.7-15.92-2.77-2.78-4.62-6.87-5-7.77-.78-2.66-2.84-5.62-6.25-5.62a5.7 5.7 0 0 0-4.6 2.46c-1-1.26-1.98-2.25-2.86-2.82A7.4 7.4 0 0 0 17.52 48Zm0 4c.51 0 1.14.22 1.82.65 2.14 1.36 6.25 8.43 7.76 11.18.5.92 1.37 1.31 2.14 1.31 1.55 0 2.75-1.53.15-3.48-3.92-2.93-2.55-7.72-.68-8.01.08-.02.17-.02.24-.02 1.7 0 2.45 2.93 2.45 2.93s2.2 5.52 5.98 9.3c3.77 3.77 3.97 6.8 1.22 10.83-1.88 2.75-5.47 3.58-9.16 3.58-3.81 0-7.73-.9-9.92-1.46-.11-.03-13.45-3.8-11.76-7 .28-.54.75-.76 1.34-.76 2.38 0 6.7 3.54 8.57 3.54.41 0 .7-.17.83-.6.79-2.85-12.06-4.05-10.98-8.17.2-.73.71-1.02 1.44-1.02 3.14 0 10.2 5.53 11.68 5.53.11 0 .2-.03.24-.1.74-1.2.33-2.04-4.9-5.2-5.21-3.16-8.88-5.06-6.8-7.33.24-.26.58-.38 1-.38 3.17 0 10.66 6.82 10.66 6.82s2.02 2.1 3.25 2.1c.28 0 .52-.1.68-.38.86-1.46-8.06-8.22-8.56-11.01-.34-1.9.24-2.85 1.31-2.85Z"></path><path fill="#FFD21E" d="M38.6 76.69c2.75-4.04 2.55-7.07-1.22-10.84-3.78-3.77-5.98-9.3-5.98-9.3s-.82-3.2-2.69-2.9c-1.87.3-3.24 5.08.68 8.01 3.91 2.93-.78 4.92-2.29 2.17-1.5-2.75-5.62-9.82-7.76-11.18-2.13-1.35-3.63-.6-3.13 2.2.5 2.79 9.43 9.55 8.56 11-.87 1.47-3.93-1.71-3.93-1.71s-9.57-8.71-11.66-6.44c-2.08 2.27 1.59 4.17 6.8 7.33 5.23 3.16 5.64 4 4.9 5.2-.75 1.2-12.28-8.53-13.36-4.4-1.08 4.11 11.77 5.3 10.98 8.15-.8 2.85-9.06-5.38-10.74-2.18-1.7 3.21 11.65 6.98 11.76 7.01 4.3 1.12 15.25 3.49 19.08-2.12Z"></path><path fill="#FF9D0B" d="M77.4 48c1.62 0 3.07.66 4.07 1.87a5.97 5.97 0 0 1 1.33 3.76 7.1 7.1 0 0 1 1.95-.3c1.55 0 2.95.59 3.94 1.66a5.8 5.8 0 0 1 .8 7 5.3 5.3 0 0 1 1.78 2.82c.24.9.48 2.8-.8 4.74a5.22 5.22 0 0 1 .37 5.02c-1.02 2.32-3.57 4.14-8.51 6.1-3.08 1.22-5.9 2-5.92 2.01a44.33 44.33 0 0 1-10.93 1.6c-5.86 0-10.05-1.8-12.46-5.34-3.88-5.69-3.33-10.9 1.7-15.92 2.78-2.78 4.63-6.87 5.01-7.77.78-2.66 2.83-5.62 6.24-5.62a5.7 5.7 0 0 1 4.6 2.46c1-1.26 1.98-2.25 2.87-2.82A7.4 7.4 0 0 1 77.4 48Zm0 4c-.51 0-1.13.22-1.82.65-2.13 1.36-6.25 8.43-7.76 11.18a2.43 2.43 0 0 1-2.14 1.31c-1.54 0-2.75-1.53-.14-3.48 3.91-2.93 2.54-7.72.67-8.01a1.54 1.54 0 0 0-.24-.02c-1.7 0-2.45 2.93-2.45 2.93s-2.2 5.52-5.97 9.3c-3.78 3.77-3.98 6.8-1.22 10.83 1.87 2.75 5.47 3.58 9.15 3.58 3.82 0 7.73-.9 9.93-1.46.1-.03 13.45-3.8 11.76-7-.29-.54-.75-.76-1.34-.76-2.38 0-6.71 3.54-8.57 3.54-.42 0-.71-.17-.83-.6-.8-2.85 12.05-4.05 10.97-8.17-.19-.73-.7-1.02-1.44-1.02-3.14 0-10.2 5.53-11.68 5.53-.1 0-.19-.03-.23-.1-.74-1.2-.34-2.04 4.88-5.2 5.23-3.16 8.9-5.06 6.8-7.33-.23-.26-.57-.38-.98-.38-3.18 0-10.67 6.82-10.67 6.82s-2.02 2.1-3.24 2.1a.74.74 0 0 1-.68-.38c-.87-1.46 8.05-8.22 8.55-11.01.34-1.9-.24-2.85-1.31-2.85Z"></path><path fill="#FFD21E" d="M56.33 76.69c-2.75-4.04-2.56-7.07 1.22-10.84 3.77-3.77 5.97-9.3 5.97-9.3s.82-3.2 2.7-2.9c1.86.3 3.23 5.08-.68 8.01-3.92 2.93.78 4.92 2.28 2.17 1.51-2.75 5.63-9.82 7.76-11.18 2.13-1.35 3.64-.6 3.13 2.2-.5 2.79-9.42 9.55-8.55 11 .86 1.47 3.92-1.71 3.92-1.71s9.58-8.71 11.66-6.44c2.08 2.27-1.58 4.17-6.8 7.33-5.23 3.16-5.63 4-4.9 5.2.75 1.2 12.28-8.53 13.36-4.4 1.08 4.11-11.76 5.3-10.97 8.15.8 2.85 9.05-5.38 10.74-2.18 1.69 3.21-11.65 6.98-11.76 7.01-4.31 1.12-15.26 3.49-19.08-2.12Z"></path></svg>

	

	<span>Datasets</span>
	

	</div></a><a class="mb-1 mr-1 md:mb-1.5 md:mr-1.5 rounded-lg" href="/datasets?library=library%3Adask"><div class="tag tag-white   "><svg class="text-black inline-block text-sm" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" fill="currentColor" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 12 12"><path d="m3.368 3.694 2.965-1.71A.095.095 0 0 0 6.38 1.9V.875a.49.49 0 0 0-.183-.394.468.468 0 0 0-.523-.034L1.526 2.84a.471.471 0 0 0-.235.408l-.002 5.405c0 .151.062.302.183.394.157.118.358.13.524.035l.878-.507a.095.095 0 0 0 .048-.082V4.465a.89.89 0 0 1 .446-.771Z" fill="#FFC11E"></path><path d="M10.475 2.919a.47.47 0 0 0-.47 0L5.856 5.312a.473.473 0 0 0-.236.408l-.002 5.425c0 .17.088.323.236.408a.466.466 0 0 0 .471 0l4.147-2.393a.473.473 0 0 0 .236-.408l.002-5.425a.467.467 0 0 0-.236-.408Z" fill="#EF1161"></path><path d="m5.647 4.949 2.737-1.58a.095.095 0 0 0 .047-.082V2.093a.49.49 0 0 0-.183-.394.468.468 0 0 0-.523-.035l-1.135.655-3.013 1.738a.471.471 0 0 0-.236.408v4.083L3.34 9.87c0 .152.062.302.183.394.157.118.358.13.524.035l1.106-.639a.094.094 0 0 0 .047-.082l.001-3.859a.89.89 0 0 1 .446-.77Z" fill="#FC6E6B"></path></svg>

	

	<span>Dask</span>
	

	</div></a><div class="relative inline-block ">
	<button class="group mr-1 mb-1 md:mr-1.5 md:mb-1.5  rounded-lg rounded-br-none " type="button">
		<div class="tag tag-white   relative rounded-br-none pr-2.5"><svg class="text-black inline-block text-sm" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" fill="none" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path d="M22.2812 12.2656L25.9532 15.7187C26.8932 16.6587 28.0913 17.3931 29.0313 16.4531C29.8594 15.7812 29.9332 14.1 29.9532 13.5C29.9532 11.5625 28.9219 8.375 27.25 6.71875C25.5604 5.04493 23.3782 3.91692 22.7032 3.78125L22.2812 12.2656Z" fill="#F5AB6A"></path><path d="M22.2812 12.2656L25.9532 15.7187C26.8932 16.6587 28.0913 17.3931 29.0313 16.4531C29.8594 15.7812 29.9332 14.1 29.9532 13.5C29.9532 11.5625 28.9219 8.375 27.25 6.71875C25.5604 5.04493 23.3782 3.91692 22.7032 3.78125L22.2812 12.2656Z" fill="url(#paint0_radial_18_31665)"></path><g filter="url(#filter0_f_18_31665)"><path d="M22.2849 12.1817L23.4375 13.2656L24.4375 4.70312C23.5121 4.1242 23.0198 3.96369 22.6563 3.89062L22.2849 12.1817Z" fill="url(#paint1_linear_18_31665)"></path></g><path d="M22.7969 3.05741L21.8437 2.65269C19.6421 1.96765 17.2344 1.81208 14.6719 2.23236C12.1094 2.65264 11.5156 3.2442 11.5156 3.2442C10.896 3.3674 10.8671 3.88898 11.1718 4.45363C13.0797 6.72576 15.176 13.4043 18.0469 14.2506C18.89 14.4662 21.3791 14.4776 22.2019 14.1593L22.3238 14.108C22.6692 13.9745 23.1672 13.2814 23.1875 12.9118L23.4219 4.03043C23.4523 3.59924 23.1484 3.25204 22.7969 3.05741Z" fill="url(#paint2_radial_18_31665)"></path><path d="M22.7969 3.05741L21.8437 2.65269C19.6421 1.96765 17.2344 1.81208 14.6719 2.23236C12.1094 2.65264 11.5156 3.2442 11.5156 3.2442C10.896 3.3674 10.8671 3.88898 11.1718 4.45363C13.0797 6.72576 15.176 13.4043 18.0469 14.2506C18.89 14.4662 21.3791 14.4776 22.2019 14.1593L22.3238 14.108C22.6692 13.9745 23.1672 13.2814 23.1875 12.9118L23.4219 4.03043C23.4523 3.59924 23.1484 3.25204 22.7969 3.05741Z" fill="url(#paint3_radial_18_31665)"></path><path d="M22.7969 3.05741L21.8437 2.65269C19.6421 1.96765 17.2344 1.81208 14.6719 2.23236C12.1094 2.65264 11.5156 3.2442 11.5156 3.2442C10.896 3.3674 10.8671 3.88898 11.1718 4.45363C13.0797 6.72576 15.176 13.4043 18.0469 14.2506C18.89 14.4662 21.3791 14.4776 22.2019 14.1593L22.3238 14.108C22.6692 13.9745 23.1672 13.2814 23.1875 12.9118L23.4219 4.03043C23.4523 3.59924 23.1484 3.25204 22.7969 3.05741Z" fill="url(#paint4_radial_18_31665)"></path><path d="M22.7969 3.05741L21.8437 2.65269C19.6421 1.96765 17.2344 1.81208 14.6719 2.23236C12.1094 2.65264 11.5156 3.2442 11.5156 3.2442C10.896 3.3674 10.8671 3.88898 11.1718 4.45363C13.0797 6.72576 15.176 13.4043 18.0469 14.2506C18.89 14.4662 21.3791 14.4776 22.2019 14.1593L22.3238 14.108C22.6692 13.9745 23.1672 13.2814 23.1875 12.9118L23.4219 4.03043C23.4523 3.59924 23.1484 3.25204 22.7969 3.05741Z" fill="url(#paint5_radial_18_31665)"></path><path d="M22.7969 3.05741L21.8437 2.65269C19.6421 1.96765 17.2344 1.81208 14.6719 2.23236C12.1094 2.65264 11.5156 3.2442 11.5156 3.2442C10.896 3.3674 10.8671 3.88898 11.1718 4.45363C13.0797 6.72576 15.176 13.4043 18.0469 14.2506C18.89 14.4662 21.3791 14.4776 22.2019 14.1593L22.3238 14.108C22.6692 13.9745 23.1672 13.2814 23.1875 12.9118L23.4219 4.03043C23.4523 3.59924 23.1484 3.25204 22.7969 3.05741Z" fill="url(#paint6_linear_18_31665)"></path><g filter="url(#filter1_f_18_31665)"><path d="M13.1016 2.72656C11.862 3.06924 11.5298 3.40016 11.5298 3.40016C10.9102 3.52335 10.936 4.11525 11.2408 4.6799C13.1487 6.95202 15.1361 13.2496 18.007 14.0958C18.2707 14.1633 18.6953 14.2107 19.1797 14.2344L13.1016 2.72656Z" fill="url(#paint7_linear_18_31665)"></path></g><path d="M12.2187 22.7187L15.7656 26.2031C16.7332 27.1171 17.3334 28.2487 16.4219 29.2188C15.7749 30.0687 14.2241 29.9933 13.625 30.0313C11.6883 30.0891 9.09014 29.5622 6.84373 27.5313C5.07737 25.9343 4.09321 23.688 3.93751 23.0156L12.2187 22.7187Z" fill="url(#paint8_radial_18_31665)"></path><path d="M12.2187 22.7187L15.7656 26.2031C16.7332 27.1171 17.3334 28.2487 16.4219 29.2188C15.7749 30.0687 14.2241 29.9933 13.625 30.0313C11.6883 30.0891 9.09014 29.5622 6.84373 27.5313C5.07737 25.9343 4.09321 23.688 3.93751 23.0156L12.2187 22.7187Z" fill="url(#paint9_radial_18_31665)"></path><g filter="url(#filter2_f_18_31665)"><path d="M12.0523 22.7916L13.2187 23.9375L4.81018 24.5721C4.4328 23.8671 4.20835 23.2768 4.14062 22.9844L12.0523 22.7916Z" fill="url(#paint10_linear_18_31665)"></path><path d="M12.0523 22.7916L13.2187 23.9375L4.81018 24.5721C4.4328 23.8671 4.20835 23.2768 4.14062 22.9844L12.0523 22.7916Z" fill="url(#paint11_radial_18_31665)"></path></g><path d="M2.99219 22.6484C2.07068 20.538 1.78913 17.4452 2.21096 15.0703C2.63279 12.6953 3.20759 11.951 3.20759 11.951C3.31231 11.3264 3.83281 11.2818 4.40626 11.5703C6.73409 13.4144 13.3119 15.2264 14.2432 18.0781C14.5947 19.3034 14.6279 21.3125 14.1641 22.5156C14.0409 22.8657 13.5 23.3516 13.0625 23.4141C12.625 23.4765 9.47656 23.3516 8.61719 23.3516C7.75781 23.3516 4.64844 23.6719 4.14062 23.6172C3.63281 23.5625 3.23437 23.2031 2.99219 22.6484Z" fill="#EC9F6A"></path><path d="M2.99219 22.6484C2.07068 20.538 1.78913 17.4452 2.21096 15.0703C2.63279 12.6953 3.20759 11.951 3.20759 11.951C3.31231 11.3264 3.83281 11.2818 4.40626 11.5703C6.73409 13.4144 13.3119 15.2264 14.2432 18.0781C14.5947 19.3034 14.6279 21.3125 14.1641 22.5156C14.0409 22.8657 13.5 23.3516 13.0625 23.4141C12.625 23.4765 9.47656 23.3516 8.61719 23.3516C7.75781 23.3516 4.64844 23.6719 4.14062 23.6172C3.63281 23.5625 3.23437 23.2031 2.99219 22.6484Z" fill="url(#paint12_radial_18_31665)"></path><path d="M2.99219 22.6484C2.07068 20.538 1.78913 17.4452 2.21096 15.0703C2.63279 12.6953 3.20759 11.951 3.20759 11.951C3.31231 11.3264 3.83281 11.2818 4.40626 11.5703C6.73409 13.4144 13.3119 15.2264 14.2432 18.0781C14.5947 19.3034 14.6279 21.3125 14.1641 22.5156C14.0409 22.8657 13.5 23.3516 13.0625 23.4141C12.625 23.4765 9.47656 23.3516 8.61719 23.3516C7.75781 23.3516 4.64844 23.6719 4.14062 23.6172C3.63281 23.5625 3.23437 23.2031 2.99219 22.6484Z" fill="url(#paint13_linear_18_31665)"></path><path d="M2.99219 22.6484C2.07068 20.538 1.78913 17.4452 2.21096 15.0703C2.63279 12.6953 3.20759 11.951 3.20759 11.951C3.31231 11.3264 3.83281 11.2818 4.40626 11.5703C6.73409 13.4144 13.3119 15.2264 14.2432 18.0781C14.5947 19.3034 14.6279 21.3125 14.1641 22.5156C14.0409 22.8657 13.5 23.3516 13.0625 23.4141C12.625 23.4765 9.47656 23.3516 8.61719 23.3516C7.75781 23.3516 4.64844 23.6719 4.14062 23.6172C3.63281 23.5625 3.23437 23.2031 2.99219 22.6484Z" fill="url(#paint14_radial_18_31665)"></path><path d="M2.99219 22.6484C2.07068 20.538 1.78913 17.4452 2.21096 15.0703C2.63279 12.6953 3.20759 11.951 3.20759 11.951C3.31231 11.3264 3.83281 11.2818 4.40626 11.5703C6.73409 13.4144 13.3119 15.2264 14.2432 18.0781C14.5947 19.3034 14.6279 21.3125 14.1641 22.5156C14.0409 22.8657 13.5 23.3516 13.0625 23.4141C12.625 23.4765 9.47656 23.3516 8.61719 23.3516C7.75781 23.3516 4.64844 23.6719 4.14062 23.6172C3.63281 23.5625 3.23437 23.2031 2.99219 22.6484Z" fill="url(#paint15_radial_18_31665)"></path><path d="M2.99219 22.6484C2.07068 20.538 1.78913 17.4452 2.21096 15.0703C2.63279 12.6953 3.20759 11.951 3.20759 11.951C3.31231 11.3264 3.83281 11.2818 4.40626 11.5703C6.73409 13.4144 13.3119 15.2264 14.2432 18.0781C14.5947 19.3034 14.6279 21.3125 14.1641 22.5156C14.0409 22.8657 13.5 23.3516 13.0625 23.4141C12.625 23.4765 9.47656 23.3516 8.61719 23.3516C7.75781 23.3516 4.64844 23.6719 4.14062 23.6172C3.63281 23.5625 3.23437 23.2031 2.99219 22.6484Z" fill="url(#paint16_radial_18_31665)"></path><g filter="url(#filter3_f_18_31665)"><path d="M2.70313 13.6719C3.04135 12.4711 3.36224 12.0555 3.36224 12.0555C3.46697 11.4309 3.98746 11.3864 4.56092 11.6749C6.88874 13.5189 13.0809 15.1104 14.0121 17.9622C14.1731 18.5231 14.2766 19.0394 14.3287 19.5128L2.70313 13.6719Z" fill="url(#paint17_linear_18_31665)"></path><path d="M2.70313 13.6719C3.04135 12.4711 3.36224 12.0555 3.36224 12.0555C3.46697 11.4309 3.98746 11.3864 4.56092 11.6749C6.88874 13.5189 13.0809 15.1104 14.0121 17.9622C14.1731 18.5231 14.2766 19.0394 14.3287 19.5128L2.70313 13.6719Z" fill="url(#paint18_linear_18_31665)"></path></g><path d="M9.83184 2.82184C6.62184 4.07184 4.07184 6.62184 2.82184 9.83184C2.48184 10.7118 2.82184 11.7018 3.62184 12.1918L14.0418 18.5418C14.6618 18.9218 15.4418 18.9218 16.0618 18.5418C17.0718 17.9218 17.9318 17.0718 18.5418 16.0618C18.9218 15.4418 18.9218 14.6618 18.5418 14.0418L12.1918 3.62184C11.7018 2.82184 10.7118 2.48184 9.83184 2.82184Z" fill="#D79453"></path><path d="M9.83184 2.82184C6.62184 4.07184 4.07184 6.62184 2.82184 9.83184C2.48184 10.7118 2.82184 11.7018 3.62184 12.1918L14.0418 18.5418C14.6618 18.9218 15.4418 18.9218 16.0618 18.5418C17.0718 17.9218 17.9318 17.0718 18.5418 16.0618C18.9218 15.4418 18.9218 14.6618 18.5418 14.0418L12.1918 3.62184C11.7018 2.82184 10.7118 2.48184 9.83184 2.82184Z" fill="url(#paint19_radial_18_31665)"></path><path d="M9.83184 2.82184C6.62184 4.07184 4.07184 6.62184 2.82184 9.83184C2.48184 10.7118 2.82184 11.7018 3.62184 12.1918L14.0418 18.5418C14.6618 18.9218 15.4418 18.9218 16.0618 18.5418C17.0718 17.9218 17.9318 17.0718 18.5418 16.0618C18.9218 15.4418 18.9218 14.6618 18.5418 14.0418L12.1918 3.62184C11.7018 2.82184 10.7118 2.48184 9.83184 2.82184Z" fill="url(#paint20_radial_18_31665)"></path><path d="M9.83184 2.82184C6.62184 4.07184 4.07184 6.62184 2.82184 9.83184C2.48184 10.7118 2.82184 11.7018 3.62184 12.1918L14.0418 18.5418C14.6618 18.9218 15.4418 18.9218 16.0618 18.5418C17.0718 17.9218 17.9318 17.0718 18.5418 16.0618C18.9218 15.4418 18.9218 14.6618 18.5418 14.0418L12.1918 3.62184C11.7018 2.82184 10.7118 2.48184 9.83184 2.82184Z" fill="url(#paint21_radial_18_31665)"></path><path d="M9.83184 2.82184C6.62184 4.07184 4.07184 6.62184 2.82184 9.83184C2.48184 10.7118 2.82184 11.7018 3.62184 12.1918L14.0418 18.5418C14.6618 18.9218 15.4418 18.9218 16.0618 18.5418C17.0718 17.9218 17.9318 17.0718 18.5418 16.0618C18.9218 15.4418 18.9218 14.6618 18.5418 14.0418L12.1918 3.62184C11.7018 2.82184 10.7118 2.48184 9.83184 2.82184Z" fill="url(#paint22_linear_18_31665)"></path><path d="M9.83184 2.82184C6.62184 4.07184 4.07184 6.62184 2.82184 9.83184C2.48184 10.7118 2.82184 11.7018 3.62184 12.1918L14.0418 18.5418C14.6618 18.9218 15.4418 18.9218 16.0618 18.5418C17.0718 17.9218 17.9318 17.0718 18.5418 16.0618C18.9218 15.4418 18.9218 14.6618 18.5418 14.0418L12.1918 3.62184C11.7018 2.82184 10.7118 2.48184 9.83184 2.82184Z" fill="url(#paint23_radial_18_31665)"></path><path d="M9.83184 2.82184C6.62184 4.07184 4.07184 6.62184 2.82184 9.83184C2.48184 10.7118 2.82184 11.7018 3.62184 12.1918L14.0418 18.5418C14.6618 18.9218 15.4418 18.9218 16.0618 18.5418C17.0718 17.9218 17.9318 17.0718 18.5418 16.0618C18.9218 15.4418 18.9218 14.6618 18.5418 14.0418L12.1918 3.62184C11.7018 2.82184 10.7118 2.48184 9.83184 2.82184Z" fill="url(#paint24_radial_18_31665)"></path><defs><filter id="filter0_f_18_31665" x="22.0349" y="3.64062" width="2.65265" height="9.875" filterUnits="userSpaceOnUse" color-interpolation-filters="sRGB"><feFlood flood-opacity="0" result="BackgroundImageFix"></feFlood><feBlend mode="normal" in="SourceGraphic" in2="BackgroundImageFix" result="shape"></feBlend><feGaussianBlur stdDeviation="0.125" result="effect1_foregroundBlur_18_31665"></feGaussianBlur></filter><filter id="filter1_f_18_31665" x="10.7815" y="2.47656" width="8.64819" height="12.0078" filterUnits="userSpaceOnUse" color-interpolation-filters="sRGB"><feFlood flood-opacity="0" result="BackgroundImageFix"></feFlood><feBlend mode="normal" in="SourceGraphic" in2="BackgroundImageFix" result="shape"></feBlend><feGaussianBlur stdDeviation="0.125" result="effect1_foregroundBlur_18_31665"></feGaussianBlur></filter><filter id="filter2_f_18_31665" x="3.89062" y="22.5416" width="9.57812" height="2.2804" filterUnits="userSpaceOnUse" color-interpolation-filters="sRGB"><feFlood flood-opacity="0" result="BackgroundImageFix"></feFlood><feBlend mode="normal" in="SourceGraphic" in2="BackgroundImageFix" result="shape"></feBlend><feGaussianBlur stdDeviation="0.125" result="effect1_foregroundBlur_18_31665"></feGaussianBlur></filter><filter id="filter3_f_18_31665" x="2.45312" y="11.2538" width="12.1255" height="8.50903" filterUnits="userSpaceOnUse" color-interpolation-filters="sRGB"><feFlood flood-opacity="0" result="BackgroundImageFix"></feFlood><feBlend mode="normal" in="SourceGraphic" in2="BackgroundImageFix" result="shape"></feBlend><feGaussianBlur stdDeviation="0.125" result="effect1_foregroundBlur_18_31665"></feGaussianBlur></filter><radialGradient id="paint0_radial_18_31665" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(22.8125 12.9375) rotate(42.7741) scale(12.5164 7.08839)"><stop offset="0.0937591" stop-color="#C05159"></stop><stop offset="0.553697" stop-color="#F6AC6A"></stop><stop offset="0.832916" stop-color="#FFD186"></stop><stop offset="0.916927" stop-color="#FFDC87"></stop></radialGradient><linearGradient id="paint1_linear_18_31665" x1="24.7344" y1="4.67187" x2="20.8594" y2="12.8906" gradientUnits="userSpaceOnUse"><stop stop-color="#EBD67C"></stop><stop offset="0.0655686" stop-color="#FFFFA6"></stop><stop offset="0.530552" stop-color="#F8C281"></stop><stop offset="0.937338" stop-color="#E99E6B"></stop></linearGradient><radialGradient id="paint2_radial_18_31665" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(22.9009 13.1847) rotate(-127.648) scale(14.3438 11.7966)"><stop stop-color="#FFBE66"></stop><stop offset="1" stop-color="#E2AE5B"></stop></radialGradient><radialGradient id="paint3_radial_18_31665" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(18 11.4375) rotate(53.9726) scale(11.9013 4.84018)"><stop stop-color="#D67C63"></stop><stop offset="1" stop-color="#D97D67" stop-opacity="0"></stop></radialGradient><radialGradient id="paint4_radial_18_31665" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(23 4.1875) rotate(45.7639) scale(3.31486 5.75622)"><stop stop-color="#FFE4A6"></stop><stop offset="0.711285" stop-color="#F8B76F"></stop><stop offset="1" stop-color="#F9B870" stop-opacity="0"></stop></radialGradient><radialGradient id="paint5_radial_18_31665" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(22.875 12.4375) rotate(88.9391) scale(3.37558 1.29066)"><stop stop-color="#FFBC67"></stop><stop offset="1" stop-color="#FFBC67" stop-opacity="0"></stop></radialGradient><linearGradient id="paint6_linear_18_31665" x1="20.375" y1="15.6875" x2="20.125" y2="12.7813" gradientUnits="userSpaceOnUse"><stop offset="0.461609" stop-color="#B45077"></stop><stop offset="0.855389" stop-color="#B75077" stop-opacity="0"></stop></linearGradient><linearGradient id="paint7_linear_18_31665" x1="12.9375" y1="2.57056" x2="18.5625" y2="14.3891" gradientUnits="userSpaceOnUse"><stop stop-color="#DDC173"></stop><stop offset="0.485173" stop-color="#D59F65"></stop><stop offset="1" stop-color="#E49966"></stop></linearGradient><radialGradient id="paint8_radial_18_31665" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(13.5625 23.5) rotate(109.113) scale(6.68078 10.2578)"><stop offset="0.165756" stop-color="#FFBF7E"></stop><stop offset="0.827674" stop-color="#DF8C6D"></stop><stop offset="1" stop-color="#B05A66"></stop></radialGradient><radialGradient id="paint9_radial_18_31665" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(15.1875 26) rotate(41.0652) scale(8.37243 2.03649)"><stop stop-color="#FFD483"></stop><stop offset="1" stop-color="#FFD688" stop-opacity="0"></stop></radialGradient><linearGradient id="paint10_linear_18_31665" x1="3.96063" y1="23.794" x2="13.3748" y2="23.5143" gradientUnits="userSpaceOnUse"><stop stop-color="#A8716F"></stop><stop offset="0.103615" stop-color="#B37173"></stop><stop offset="0.225484" stop-color="#DB9F84"></stop><stop offset="0.799889" stop-color="#F1BB8A"></stop><stop offset="1" stop-color="#FFD780"></stop></linearGradient><radialGradient id="paint11_radial_18_31665" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(11.4219 23.1719) rotate(-178.616) scale(3.23532 0.569081)"><stop offset="0.621498" stop-color="#AF5A3E"></stop><stop offset="1" stop-color="#B35445" stop-opacity="0"></stop></radialGradient><radialGradient id="paint12_radial_18_31665" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(13.625 19.125) rotate(-171.737) scale(15.2205 15.0767)"><stop offset="0.138435" stop-color="#FFB974"></stop><stop offset="0.403618" stop-color="#F2A56D"></stop><stop offset="0.925938" stop-color="#A16948"></stop></radialGradient><linearGradient id="paint13_linear_18_31665" x1="8.22184" y1="13.125" x2="6.81191" y2="15.4996" gradientUnits="userSpaceOnUse"><stop offset="0.610751" stop-color="#984847"></stop><stop offset="0.850075" stop-color="#9A4947" stop-opacity="0"></stop></linearGradient><radialGradient id="paint14_radial_18_31665" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(7.25 23.7461) scale(11.25 5.68361)"><stop stop-color="#C66364"></stop><stop offset="1" stop-color="#D4766B" stop-opacity="0"></stop></radialGradient><radialGradient id="paint15_radial_18_31665" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(7.875 23.5313) scale(10.0937 1.29657)"><stop stop-color="#B64B4B"></stop><stop offset="1" stop-color="#C56158" stop-opacity="0"></stop></radialGradient><radialGradient id="paint16_radial_18_31665" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(11.4375 19.875) rotate(-46.8882) scale(4.02385 7.51767)"><stop stop-color="#FFC083"></stop><stop offset="0.620218" stop-color="#FFBD7D" stop-opacity="0"></stop></radialGradient><linearGradient id="paint17_linear_18_31665" x1="2.8125" y1="13.0312" x2="14.5582" y2="18.9404" gradientUnits="userSpaceOnUse"><stop stop-color="#B89367"></stop><stop offset="1" stop-color="#C5835E"></stop></linearGradient><linearGradient id="paint18_linear_18_31665" x1="8.21875" y1="14.6406" x2="7.59349" y2="15.6717" gradientUnits="userSpaceOnUse"><stop offset="0.351552" stop-color="#A74746"></stop><stop offset="0.845198" stop-color="#A04346" stop-opacity="0"></stop></linearGradient><radialGradient id="paint19_radial_18_31665" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(15.5625 14.5625) rotate(140.244) scale(18.3733 13.7403)"><stop stop-color="#FDAE69"></stop><stop offset="0.729021" stop-color="#CE8C4F"></stop><stop offset="0.921546" stop-color="#AD7B45"></stop><stop offset="1" stop-color="#8B6B4A"></stop></radialGradient><radialGradient id="paint20_radial_18_31665" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(15.0625 7) rotate(65.3152) scale(11.0745 3.16547)"><stop offset="0.233237" stop-color="#FFD47C"></stop><stop offset="0.853648" stop-color="#FFD98B" stop-opacity="0"></stop></radialGradient><radialGradient id="paint21_radial_18_31665" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(15.3125 8.875) rotate(100.886) scale(6.6191 5.57808)"><stop offset="0.128419" stop-color="#FFD88C"></stop><stop offset="0.924134" stop-color="#FFBE7B" stop-opacity="0"></stop></radialGradient><linearGradient id="paint22_linear_18_31665" x1="7.25" y1="15.1875" x2="10.7588" y2="10.3142" gradientUnits="userSpaceOnUse"><stop offset="0.142353" stop-color="#C15F4D"></stop><stop offset="1" stop-color="#D58366" stop-opacity="0"></stop></linearGradient><radialGradient id="paint23_radial_18_31665" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(8.15625 15.7813) rotate(28.5422) scale(12.5574 1.96589)"><stop offset="0.149989" stop-color="#E4745D"></stop><stop offset="0.453292" stop-color="#C8604C"></stop><stop offset="0.632597" stop-color="#C0605F"></stop><stop offset="1" stop-color="#C0605F" stop-opacity="0"></stop></radialGradient><radialGradient id="paint24_radial_18_31665" cx="0" cy="0" r="1" gradientUnits="userSpaceOnUse" gradientTransform="translate(1.40625 2.69067) rotate(46.0943) scale(22.3963)"><stop offset="0.935802" stop-color="#C17C61" stop-opacity="0"></stop><stop offset="0.982109" stop-color="#C17C61"></stop></radialGradient></defs></svg>

	

	<span>Croissant</span>
	

	<div class="border-br-gray-200 absolute bottom-0.5 right-0.5 h-1 w-1 border-[3px] border-l-transparent border-t-transparent border-b-gray-200 border-r-gray-200 dark:border-b-gray-700 dark:border-r-gray-700"></div></div>
		
		</button>
	
	
	</div>

	</div></div>

		<div class="flex flex-col-reverse lg:flex-row lg:items-center lg:justify-between"><div class="-mb-px flex h-12 items-center overflow-x-auto overflow-y-hidden "><a class="tab-alternate " href="/datasets/BAAI/CCI3-HQ"><svg class="mr-1.5 text-gray-400 flex-none" style="" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 24 24"><path class="uim-quaternary" d="M20.23 7.24L12 12L3.77 7.24a1.98 1.98 0 0 1 .7-.71L11 2.76c.62-.35 1.38-.35 2 0l6.53 3.77c.29.173.531.418.7.71z" opacity=".25" fill="currentColor"></path><path class="uim-tertiary" d="M12 12v9.5a2.09 2.09 0 0 1-.91-.21L4.5 17.48a2.003 2.003 0 0 1-1-1.73v-7.5a2.06 2.06 0 0 1 .27-1.01L12 12z" opacity=".5" fill="currentColor"></path><path class="uim-primary" d="M20.5 8.25v7.5a2.003 2.003 0 0 1-1 1.73l-6.62 3.82c-.275.13-.576.198-.88.2V12l8.23-4.76c.175.308.268.656.27 1.01z" fill="currentColor"></path></svg>
			Dataset card
			
			
		</a><a class="tab-alternate " href="/datasets/BAAI/CCI3-HQ/viewer/"><svg class="mr-1.5 text-gray-400 flex-none" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 12 12"><path fill="currentColor" d="M2.5 2h7a1 1 0 0 1 1 1v6a1 1 0 0 1-1 1h-7a1 1 0 0 1-1-1V3a1 1 0 0 1 1-1Zm0 2v2h3V4h-3Zm4 0v2h3V4h-3Zm-4 3v2h3V7h-3Zm4 0v2h3V7h-3Z"></path></svg>
			Viewer
			
			
		</a><a class="tab-alternate active" href="/datasets/BAAI/CCI3-HQ/tree/main"><svg class="mr-1.5 text-gray-400 flex-none" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 24 24"><path class="uim-tertiary" d="M21 19h-8a1 1 0 0 1 0-2h8a1 1 0 0 1 0 2zm0-4h-8a1 1 0 0 1 0-2h8a1 1 0 0 1 0 2zm0-8h-8a1 1 0 0 1 0-2h8a1 1 0 0 1 0 2zm0 4h-8a1 1 0 0 1 0-2h8a1 1 0 0 1 0 2z" opacity=".5" fill="currentColor"></path><path class="uim-primary" d="M9 19a1 1 0 0 1-1-1V6a1 1 0 0 1 2 0v12a1 1 0 0 1-1 1zm-6-4.333a1 1 0 0 1-.64-1.769L3.438 12l-1.078-.898a1 1 0 0 1 1.28-1.538l2 1.667a1 1 0 0 1 0 1.538l-2 1.667a.999.999 0 0 1-.64.231z" fill="currentColor"></path></svg>
			<span class="xl:hidden">Files</span>
				<span class="hidden xl:inline">Files and versions</span>
			
			
		</a><a class="tab-alternate " href="/datasets/BAAI/CCI3-HQ/discussions"><svg class="mr-1.5 text-gray-400 flex-none" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path d="M20.6081 3C21.7684 3 22.8053 3.49196 23.5284 4.38415C23.9756 4.93678 24.4428 5.82749 24.4808 7.16133C24.9674 7.01707 25.4353 6.93643 25.8725 6.93643C26.9833 6.93643 27.9865 7.37587 28.696 8.17411C29.6075 9.19872 30.0124 10.4579 29.8361 11.7177C29.7523 12.3177 29.5581 12.8555 29.2678 13.3534C29.8798 13.8646 30.3306 14.5763 30.5485 15.4322C30.719 16.1032 30.8939 17.5006 29.9808 18.9403C30.0389 19.0342 30.0934 19.1319 30.1442 19.2318C30.6932 20.3074 30.7283 21.5229 30.2439 22.6548C29.5093 24.3704 27.6841 25.7219 24.1397 27.1727C21.9347 28.0753 19.9174 28.6523 19.8994 28.6575C16.9842 29.4379 14.3477 29.8345 12.0653 29.8345C7.87017 29.8345 4.8668 28.508 3.13831 25.8921C0.356375 21.6797 0.754104 17.8269 4.35369 14.1131C6.34591 12.058 7.67023 9.02782 7.94613 8.36275C8.50224 6.39343 9.97271 4.20438 12.4172 4.20438H12.4179C12.6236 4.20438 12.8314 4.2214 13.0364 4.25468C14.107 4.42854 15.0428 5.06476 15.7115 6.02205C16.4331 5.09583 17.134 4.359 17.7682 3.94323C18.7242 3.31737 19.6794 3 20.6081 3ZM20.6081 5.95917C20.2427 5.95917 19.7963 6.1197 19.3039 6.44225C17.7754 7.44319 14.8258 12.6772 13.7458 14.7131C13.3839 15.3952 12.7655 15.6837 12.2086 15.6837C11.1036 15.6837 10.2408 14.5497 12.1076 13.1085C14.9146 10.9402 13.9299 7.39584 12.5898 7.1776C12.5311 7.16799 12.4731 7.16355 12.4172 7.16355C11.1989 7.16355 10.6615 9.33114 10.6615 9.33114C10.6615 9.33114 9.0863 13.4148 6.38031 16.206C3.67434 18.998 3.5346 21.2388 5.50675 24.2246C6.85185 26.2606 9.42666 26.8753 12.0653 26.8753C14.8021 26.8753 17.6077 26.2139 19.1799 25.793C19.2574 25.7723 28.8193 22.984 27.6081 20.6107C27.4046 20.212 27.0693 20.0522 26.6471 20.0522C24.9416 20.0522 21.8393 22.6726 20.5057 22.6726C20.2076 22.6726 19.9976 22.5416 19.9116 22.222C19.3433 20.1173 28.552 19.2325 27.7758 16.1839C27.639 15.6445 27.2677 15.4256 26.746 15.4263C24.4923 15.4263 19.4358 19.5181 18.3759 19.5181C18.2949 19.5181 18.2368 19.4937 18.2053 19.4419C17.6743 18.557 17.9653 17.9394 21.7082 15.6009C25.4511 13.2617 28.0783 11.8545 26.5841 10.1752C26.4121 9.98141 26.1684 9.8956 25.8725 9.8956C23.6001 9.89634 18.2311 14.9403 18.2311 14.9403C18.2311 14.9403 16.7821 16.496 15.9057 16.496C15.7043 16.496 15.533 16.4139 15.4169 16.2112C14.7956 15.1296 21.1879 10.1286 21.5484 8.06535C21.7928 6.66715 21.3771 5.95917 20.6081 5.95917Z" fill="#FF9D00"></path><path d="M5.50686 24.2246C3.53472 21.2387 3.67446 18.9979 6.38043 16.206C9.08641 13.4147 10.6615 9.33111 10.6615 9.33111C10.6615 9.33111 11.2499 6.95933 12.59 7.17757C13.93 7.39581 14.9139 10.9401 12.1069 13.1084C9.29997 15.276 12.6659 16.7489 13.7459 14.713C14.8258 12.6772 17.7747 7.44316 19.304 6.44221C20.8326 5.44128 21.9089 6.00204 21.5484 8.06532C21.188 10.1286 14.795 15.1295 15.4171 16.2118C16.0391 17.2934 18.2312 14.9402 18.2312 14.9402C18.2312 14.9402 25.0907 8.49588 26.5842 10.1752C28.0776 11.8545 25.4512 13.2616 21.7082 15.6008C17.9646 17.9393 17.6744 18.557 18.2054 19.4418C18.7372 20.3266 26.9998 13.1351 27.7759 16.1838C28.5513 19.2324 19.3434 20.1173 19.9117 22.2219C20.48 24.3274 26.3979 18.2382 27.6082 20.6107C28.8193 22.9839 19.2574 25.7722 19.18 25.7929C16.0914 26.62 8.24723 28.3726 5.50686 24.2246Z" fill="#FFD21E"></path></svg>
			Community
			<div class="ml-1.5 flex h-4 min-w-[1rem] items-center justify-center rounded px-1 text-xs leading-none shadow-sm bg-black text-white dark:bg-gray-800 dark:text-gray-200">1
				</div>
			
		</a>
	</div>
			</div></div></header></div>
	
<div class="container relative flex flex-col md:grid md:space-y-0 w-full md:grid-cols-12  space-y-4 md:gap-6 mb-16"><section class="pt-8 border-gray-100 col-span-full"><header class="flex flex-wrap items-center justify-start pb-2 md:justify-end lg:flex-nowrap"><div class="relative mr-4 flex min-w-0 basis-auto flex-wrap items-center md:flex-grow md:basis-full lg:basis-auto lg:flex-nowrap"><div class="SVELTE_HYDRATER contents" data-target="BranchSelector" data-props="{&quot;path&quot;:&quot;lighteval_tasks_v2.py&quot;,&quot;repoName&quot;:&quot;BAAI/CCI3-HQ&quot;,&quot;repoType&quot;:&quot;dataset&quot;,&quot;rev&quot;:&quot;main&quot;,&quot;refs&quot;:{&quot;branches&quot;:[{&quot;name&quot;:&quot;main&quot;,&quot;ref&quot;:&quot;refs/heads/main&quot;,&quot;targetCommit&quot;:&quot;b2a6a0eae226a8d877d62c414a890028d3c6f507&quot;}],&quot;tags&quot;:[],&quot;converts&quot;:[{&quot;name&quot;:&quot;duckdb&quot;,&quot;ref&quot;:&quot;refs/convert/duckdb&quot;,&quot;targetCommit&quot;:&quot;2053371d111adabe3988f2e286f6b106ddfeff7d&quot;},{&quot;name&quot;:&quot;parquet&quot;,&quot;ref&quot;:&quot;refs/convert/parquet&quot;,&quot;targetCommit&quot;:&quot;764b5e7fddc1ac261b09967a90038b69a12b4cf5&quot;}]},&quot;view&quot;:&quot;blob&quot;}"><div class="relative mr-4 mb-2">
	<button class="text-sm md:text-base btn w-full cursor-pointer text-sm" type="button">
		<svg class="mr-1.5 text-gray-700 dark:text-gray-400" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 24 24" style="transform: rotate(360deg);"><path d="M13 14c-3.36 0-4.46 1.35-4.82 2.24C9.25 16.7 10 17.76 10 19a3 3 0 0 1-3 3a3 3 0 0 1-3-3c0-1.31.83-2.42 2-2.83V7.83A2.99 2.99 0 0 1 4 5a3 3 0 0 1 3-3a3 3 0 0 1 3 3c0 1.31-.83 2.42-2 2.83v5.29c.88-.65 2.16-1.12 4-1.12c2.67 0 3.56-1.34 3.85-2.23A3.006 3.006 0 0 1 14 7a3 3 0 0 1 3-3a3 3 0 0 1 3 3c0 1.34-.88 2.5-2.09 2.86C17.65 11.29 16.68 14 13 14m-6 4a1 1 0 0 0-1 1a1 1 0 0 0 1 1a1 1 0 0 0 1-1a1 1 0 0 0-1-1M7 4a1 1 0 0 0-1 1a1 1 0 0 0 1 1a1 1 0 0 0 1-1a1 1 0 0 0-1-1m10 2a1 1 0 0 0-1 1a1 1 0 0 0 1 1a1 1 0 0 0 1-1a1 1 0 0 0-1-1z" fill="currentColor"></path></svg>
			main
		<svg class="-mr-1 text-gray-500" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 24 24"><path d="M16.293 9.293L12 13.586L7.707 9.293l-1.414 1.414L12 16.414l5.707-5.707z" fill="currentColor"></path></svg></button>
	
	
	</div></div>
		<div class="relative mb-2 flex flex-wrap items-center"><a class="truncate text-gray-800 hover:underline" href="/datasets/BAAI/CCI3-HQ/tree/main">CCI3-HQ</a>
			<span class="mx-1 text-gray-300">/</span>
				<span class="dark:text-gray-300">lighteval_tasks_v2.py</span>
				<div class="SVELTE_HYDRATER contents" data-target="CopyButton" data-props="{&quot;value&quot;:&quot;lighteval_tasks_v2.py&quot;,&quot;classNames&quot;:&quot;text-xs ml-2&quot;,&quot;title&quot;:&quot;Copy path&quot;}"><button class="relative text-xs ml-2 inline-flex cursor-pointer items-center text-sm focus:outline-none  mx-0.5   text-gray-600 " title="Copy path" type="button"><svg class="" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" fill="currentColor" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path d="M28,10V28H10V10H28m0-2H10a2,2,0,0,0-2,2V28a2,2,0,0,0,2,2H28a2,2,0,0,0,2-2V10a2,2,0,0,0-2-2Z" transform="translate(0)"></path><path d="M4,18H2V4A2,2,0,0,1,4,2H18V4H4Z" transform="translate(0)"></path><rect fill="none" width="32" height="32"></rect></svg>
	
	</button></div></div></div>

	
	</header>
			<div class="SVELTE_HYDRATER contents" data-target="LastCommit" data-props="{&quot;commitLast&quot;:{&quot;date&quot;:&quot;2024-09-26T08:24:16.000Z&quot;,&quot;verified&quot;:&quot;verified&quot;,&quot;subject&quot;:&quot;Create lighteval_tasks_v2.py&quot;,&quot;authors&quot;:[{&quot;_id&quot;:&quot;63a11ce02fabbbb899a01d58&quot;,&quot;avatar&quot;:&quot;/avatars/ee3d4088b6d32b2c18b8be91913e90dd.svg&quot;,&quot;isHf&quot;:false,&quot;user&quot;:&quot;ldwang&quot;}],&quot;commit&quot;:{&quot;id&quot;:&quot;b2a6a0eae226a8d877d62c414a890028d3c6f507&quot;,&quot;parentIds&quot;:[&quot;f9bb8f392b6b7a292feeedc2a90ac11ff47ef4c8&quot;]},&quot;title&quot;:&quot;Create lighteval_tasks_v2.py&quot;},&quot;repo&quot;:{&quot;name&quot;:&quot;BAAI/CCI3-HQ&quot;,&quot;type&quot;:&quot;dataset&quot;}}"><div class="from-gray-100-to-white flex items-baseline rounded-t-lg border border-b-0 bg-gradient-to-t px-3 py-2 dark:border-gray-800"><img class="mr-2.5 mt-0.5 h-4 w-4 self-center rounded-full" alt="ldwang's picture" src="/avatars/ee3d4088b6d32b2c18b8be91913e90dd.svg">
			<div class="mr-5 flex flex-none items-center truncate"><a class="hover:underline" href="/ldwang">ldwang
					</a>
				
			</div>
		<div class="mr-4 truncate font-mono text-sm text-gray-500 hover:prose-a:underline"><!-- HTML_TAG_START -->Create lighteval_tasks_v2.py<!-- HTML_TAG_END --></div>
		<a class="rounded border bg-gray-50 px-1.5 text-sm hover:underline dark:border-gray-800 dark:bg-gray-900" href="/datasets/BAAI/CCI3-HQ/commit/b2a6a0eae226a8d877d62c414a890028d3c6f507">b2a6a0e</a>
		<span class="mx-2 text-green-500 dark:text-green-600 px-1.5 border-green-100 dark:border-green-800 rounded-full border text-xs uppercase" title="This commit is signed and the signature is verified">verified</span>
		<time class="ml-auto hidden flex-none truncate pl-2 text-gray-500 dark:text-gray-400 lg:block" datetime="2024-09-26T08:24:16" title="Thu, 26 Sep 2024 08:24:16 GMT">3 days ago</time></div></div>
			<div class="flex flex-wrap items-center border px-3 py-1.5 text-sm text-gray-800 dark:border-gray-800 dark:bg-gray-900">
				<a class="my-1 mr-4 flex items-center hover:underline " href="/datasets/BAAI/CCI3-HQ/raw/main/lighteval_tasks_v2.py"><svg class="mr-1.5" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32" style="transform: rotate(360deg);"><path d="M31 16l-7 7l-1.41-1.41L28.17 16l-5.58-5.59L24 9l7 7z" fill="currentColor"></path><path d="M1 16l7-7l1.41 1.41L3.83 16l5.58 5.59L8 23l-7-7z" fill="currentColor"></path><path d="M12.419 25.484L17.639 6l1.932.518L14.35 26z" fill="currentColor"></path></svg>
							raw
						</a><div class="SVELTE_HYDRATER contents" data-target="CopyButton" data-props="{&quot;value&quot;:&quot;https://huggingface.co/datasets/BAAI/CCI3-HQ/resolve/main/lighteval_tasks_v2.py&quot;,&quot;style&quot;:&quot;blank&quot;,&quot;label&quot;:&quot;Copy download link&quot;,&quot;classNames&quot;:&quot;my-1 mr-4 flex items-center no-underline hover:underline&quot;}"><button class="relative my-1 mr-4 flex items-center no-underline hover:underline       " title="Copy download link" type="button"><svg class="" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" fill="currentColor" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path d="M28,10V28H10V10H28m0-2H10a2,2,0,0,0-2,2V28a2,2,0,0,0,2,2H28a2,2,0,0,0,2-2V10a2,2,0,0,0-2-2Z" transform="translate(0)"></path><path d="M4,18H2V4A2,2,0,0,1,4,2H18V4H4Z" transform="translate(0)"></path><rect fill="none" width="32" height="32"></rect></svg>
	<span class="ml-1.5 ">Copy download link</span>
	</button></div><a class="my-1 mr-4 flex items-center hover:underline " href="/datasets/BAAI/CCI3-HQ/commits/main/lighteval_tasks_v2.py"><svg class="mr-1.5" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32" style="transform: rotate(360deg);"><path d="M16 4C9.383 4 4 9.383 4 16s5.383 12 12 12s12-5.383 12-12S22.617 4 16 4zm0 2c5.535 0 10 4.465 10 10s-4.465 10-10 10S6 21.535 6 16S10.465 6 16 6zm-1 2v9h7v-2h-5V8z" fill="currentColor"></path></svg>
							history
						</a><a class="my-1 mr-4 flex items-center hover:underline " href="/datasets/BAAI/CCI3-HQ/blame/main/lighteval_tasks_v2.py"><svg class="mr-1.5" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32" style="transform: rotate(360deg);"><path d="M16 2a14 14 0 1 0 14 14A14 14 0 0 0 16 2zm0 26a12 12 0 1 1 12-12a12 12 0 0 1-12 12z" fill="currentColor"></path><path d="M11.5 11a2.5 2.5 0 1 0 2.5 2.5a2.48 2.48 0 0 0-2.5-2.5z" fill="currentColor"></path><path d="M20.5 11a2.5 2.5 0 1 0 2.5 2.5a2.48 2.48 0 0 0-2.5-2.5z" fill="currentColor"></path></svg>
							blame
						</a><a class="my-1 mr-4 flex items-center hover:underline text-green-600 dark:text-gray-300" href="/datasets/BAAI/CCI3-HQ/edit/main/lighteval_tasks_v2.py"><svg class="mr-1.5" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path d="M2 26h28v2H2z" fill="currentColor"></path><path d="M25.4 9c.8-.8.8-2 0-2.8l-3.6-3.6c-.8-.8-2-.8-2.8 0l-15 15V24h6.4l15-15zm-5-5L24 7.6l-3 3L17.4 7l3-3zM6 22v-3.6l10-10l3.6 3.6l-10 10H6z" fill="currentColor"></path></svg>
							contribute
						</a><a class="my-1 mr-4 flex items-center hover:underline " href="/datasets/BAAI/CCI3-HQ/delete/main/lighteval_tasks_v2.py"><svg class="mr-1.5" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 32 32"><path d="M12 12h2v12h-2z" fill="currentColor"></path><path d="M18 12h2v12h-2z" fill="currentColor"></path><path d="M4 6v2h2v20a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V8h2V6zm4 22V8h16v20z" fill="currentColor"></path><path d="M12 2h8v2h-8z" fill="currentColor"></path></svg>
							delete
						</a>
				<div class="mr-4 flex items-center text-gray-400"><svg class="text-gray-300 text-sm mr-1.5 -translate-y-px" width="1em" height="1em" viewBox="0 0 22 28" fill="none" xmlns="http://www.w3.org/2000/svg"><path fill-rule="evenodd" clip-rule="evenodd" d="M15.3634 10.3639C15.8486 10.8491 15.8486 11.6357 15.3634 12.1209L10.9292 16.5551C10.6058 16.8785 10.0814 16.8785 9.7579 16.5551L7.03051 13.8277C6.54532 13.3425 6.54532 12.5558 7.03051 12.0707C7.51569 11.5855 8.30234 11.5855 8.78752 12.0707L9.7579 13.041C10.0814 13.3645 10.6058 13.3645 10.9292 13.041L13.6064 10.3639C14.0916 9.8787 14.8782 9.8787 15.3634 10.3639Z" fill="currentColor"></path><path fill-rule="evenodd" clip-rule="evenodd" d="M10.6666 27.12C4.93329 25.28 0 19.2267 0 12.7867V6.52001C0 5.40001 0.693334 4.41334 1.73333 4.01334L9.73333 1.01334C10.3333 0.786673 11 0.786673 11.6 1.02667L19.6 4.02667C20.1083 4.21658 20.5465 4.55701 20.8562 5.00252C21.1659 5.44803 21.3324 5.97742 21.3333 6.52001V12.7867C21.3333 19.24 16.4 25.28 10.6666 27.12Z" fill="currentColor" fill-opacity="0.22"></path><path d="M10.0845 1.94967L10.0867 1.94881C10.4587 1.8083 10.8666 1.81036 11.2286 1.95515L11.2387 1.95919L11.2489 1.963L19.2489 4.963L19.25 4.96342C19.5677 5.08211 19.8416 5.29488 20.0351 5.57333C20.2285 5.85151 20.3326 6.18203 20.3333 6.52082C20.3333 6.52113 20.3333 6.52144 20.3333 6.52176L20.3333 12.7867C20.3333 18.6535 15.8922 24.2319 10.6666 26.0652C5.44153 24.2316 1 18.6409 1 12.7867V6.52001C1 5.82357 1.42893 5.20343 2.08883 4.94803L10.0845 1.94967Z" stroke="currentColor" stroke-opacity="0.30" stroke-width="2"></path></svg>

							No virus
						</div>
				
				<div class="flex items-center gap-x-3 dark:text-gray-300 sm:ml-auto"><div class="SVELTE_HYDRATER contents" data-target="LineWrapButton" data-props="{&quot;classNames&quot;:&quot;text-xs&quot;,&quot;lineSelectorClass&quot;:&quot;blob-line&quot;}">

<button class="text-xs" type="button" title="Toggle Line Wrap"><svg class="opacity-50" width="1em" height="1em" viewBox="0 0 12 11" fill="none" xmlns="http://www.w3.org/2000/svg"><path d="M0.75 1.25H11.25M0.75 5H9C9.75 5 11.25 5.375 11.25 6.875C11.25 8.375 9.99975 8.75 9.375 8.75H6M6 8.75L7.5 7.25M6 8.75L7.5 10.25M0.75 8.75H3.75" stroke="currentColor" stroke-width="1.125" stroke-linecap="round" stroke-linejoin="round"></path></svg></button></div>
					42.1 kB</div></div>

			<div class="relative min-h-[100px] rounded-b-lg border border-t-0 leading-tight dark:border-gray-800 dark:bg-gray-925">
				<div class="py-3"><div class="SVELTE_HYDRATER contents" data-target="BlobContent" data-props="{&quot;lines&quot;:[&quot;<span class=\&quot;hljs-comment\&quot;># ruff: noqa: F405, F403, F401</span>&quot;,&quot;<span class=\&quot;hljs-string\&quot;>&amp;quot;&amp;quot;&amp;quot;</span>&quot;,&quot;<span class=\&quot;hljs-string\&quot;>Custom evaluation tasks for lighteval</span>&quot;,&quot;<span class=\&quot;hljs-string\&quot;></span>&quot;,&quot;<span class=\&quot;hljs-string\&quot;>Do note that we ran the evals with `max_samples=1000` to speed up large evals.</span>&quot;,&quot;<span class=\&quot;hljs-string\&quot;>Most custom prompt changes were in an attempt to improve signal for small models in general.</span>&quot;,&quot;<span class=\&quot;hljs-string\&quot;></span>&quot;,&quot;<span class=\&quot;hljs-string\&quot;>This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.</span>&quot;,&quot;<span class=\&quot;hljs-string\&quot;></span>&quot;,&quot;<span class=\&quot;hljs-string\&quot;>Example usage (lighteval_tasks.py is the path to this file):</span>&quot;,&quot;<span class=\&quot;hljs-string\&quot;>===================</span>&quot;,&quot;<span class=\&quot;hljs-string\&quot;>accelerate launch --num_processes=1 lighteval/run_evals_accelerate.py --model_args=&amp;quot;pretrained=HuggingFaceFW/ablation-model-fineweb-edu&amp;quot; \\</span>&quot;,&quot;<span class=\&quot;hljs-string\&quot;>    --custom_tasks &amp;quot;lighteval_tasks.py&amp;quot; --output_dir [OUTPUTPATH] --max_samples 1000 \\ </span>&quot;,&quot;<span class=\&quot;hljs-string\&quot;>    --tasks &amp;quot;custom|hellaswag|0|1,custom|winogrande|0|1,custom|piqa|0|1,custom|siqa|0|1,custom|openbookqa|0|1,custom|arc:easy|0|1,custom|arc:challenge|0|1,custom|commonsense_qa|0|1,custom|mmlu:abstract_algebra|0|1,custom|mmlu:anatomy|0|1,custom|mmlu:astronomy|0|1,custom|mmlu:business_ethics|0|1,custom|mmlu:clinical_knowledge|0|1,custom|mmlu:college_biology|0|1,custom|mmlu:college_chemistry|0|1,custom|mmlu:college_computer_science|0|1,custom|mmlu:college_mathematics|0|1,custom|mmlu:college_medicine|0|1,custom|mmlu:college_physics|0|1,custom|mmlu:computer_security|0|1,custom|mmlu:conceptual_physics|0|1,custom|mmlu:econometrics|0|1,custom|mmlu:electrical_engineering|0|1,custom|mmlu:elementary_mathematics|0|1,custom|mmlu:formal_logic|0|1,custom|mmlu:global_facts|0|1,custom|mmlu:high_school_biology|0|1,custom|mmlu:high_school_chemistry|0|1,custom|mmlu:high_school_computer_science|0|1,custom|mmlu:high_school_european_history|0|1,custom|mmlu:high_school_geography|0|1,custom|mmlu:high_school_government_and_politics|0|1,custom|mmlu:high_school_macroeconomics|0|1,custom|mmlu:high_school_mathematics|0|1,custom|mmlu:high_school_microeconomics|0|1,custom|mmlu:high_school_physics|0|1,custom|mmlu:high_school_psychology|0|1,custom|mmlu:high_school_statistics|0|1,custom|mmlu:high_school_us_history|0|1,custom|mmlu:high_school_world_history|0|1,custom|mmlu:human_aging|0|1,custom|mmlu:human_sexuality|0|1,custom|mmlu:international_law|0|1,custom|mmlu:jurisprudence|0|1,custom|mmlu:logical_fallacies|0|1,custom|mmlu:machine_learning|0|1,custom|mmlu:management|0|1,custom|mmlu:marketing|0|1,custom|mmlu:medical_genetics|0|1,custom|mmlu:miscellaneous|0|1,custom|mmlu:moral_disputes|0|1,custom|mmlu:moral_scenarios|0|1,custom|mmlu:nutrition|0|1,custom|mmlu:philosophy|0|1,custom|mmlu:prehistory|0|1,custom|mmlu:professional_accounting|0|1,custom|mmlu:professional_law|0|1,custom|mmlu:professional_medicine|0|1,custom|mmlu:professional_psychology|0|1,custom|mmlu:public_relations|0|1,custom|mmlu:security_studies|0|1,custom|mmlu:sociology|0|1,custom|mmlu:us_foreign_policy|0|1,custom|mmlu:virology|0|1,custom|mmlu:world_religions|0|1&amp;quot;</span>&quot;,&quot;<span class=\&quot;hljs-string\&quot;>===================</span>&quot;,&quot;<span class=\&quot;hljs-string\&quot;>custom|cmmlu:agronomy|0|1,custom|cmmlu:anatomy|0|1,custom|cmmlu:ancient_chinese|0|1,custom|cmmlu:arts|0|1,custom|cmmlu:astronomy|0|1,custom|cmmlu:business_ethics|0|1,custom|cmmlu:chinese_civil_service_exam|0|1,custom|cmmlu:chinese_driving_rule|0|1,custom|cmmlu:chinese_food_culture|0|1,custom|cmmlu:chinese_foreign_policy|0|1,custom|cmmlu:chinese_history|0|1,custom|cmmlu:chinese_literature|0|1,custom|cmmlu:chinese_teacher_qualification|0|1,custom|cmmlu:clinical_knowledge|0|1,custom|cmmlu:college_actuarial_science|0|1,custom|cmmlu:college_education|0|1,custom|cmmlu:college_engineering_hydrology|0|1,custom|cmmlu:college_law|0|1,custom|cmmlu:college_mathematics|0|1,custom|cmmlu:college_medical_statistics|0|1,custom|cmmlu:college_medicine|0|1,custom|cmmlu:computer_science|0|1,custom|cmmlu:computer_security|0|1,custom|cmmlu:conceptual_physics|0|1,custom|cmmlu:construction_project_management|0|1,custom|cmmlu:economics|0|1,custom|cmmlu:education|0|1,custom|cmmlu:electrical_engineering|0|1,custom|cmmlu:elementary_chinese|0|1,custom|cmmlu:elementary_commonsense|0|1,custom|cmmlu:elementary_information_and_technology|0|1,custom|cmmlu:elementary_mathematics|0|1,custom|cmmlu:ethnology|0|1,custom|cmmlu:food_science|0|1,custom|cmmlu:genetics|0|1,custom|cmmlu:global_facts|0|1,custom|cmmlu:high_school_biology|0|1,custom|cmmlu:high_school_chemistry|0|1,custom|cmmlu:high_school_geography|0|1,custom|cmmlu:high_school_mathematics|0|1,custom|cmmlu:high_school_physics|0|1,custom|cmmlu:high_school_politics|0|1,custom|cmmlu:human_sexuality|0|1,custom|cmmlu:international_law|0|1,custom|cmmlu:journalism|0|1,custom|cmmlu:jurisprudence|0|1,custom|cmmlu:legal_and_moral_basis|0|1,custom|cmmlu:logical|0|1,custom|cmmlu:machine_learning|0|1,custom|cmmlu:management|0|1,custom|cmmlu:marketing|0|1,custom|cmmlu:marxist_theory|0|1,custom|cmmlu:modern_chinese|0|1,custom|cmmlu:nutrition|0|1,custom|cmmlu:philosophy|0|1,custom|cmmlu:professional_accounting|0|1,custom|cmmlu:professional_law|0|1,custom|cmmlu:professional_medicine|0|1,custom|cmmlu:professional_psychology|0|1,custom|cmmlu:public_relations|0|1,custom|cmmlu:security_study|0|1,custom|cmmlu:sociology|0|1,custom|cmmlu:sports_science|0|1,custom|cmmlu:traditional_chinese_medicine|0|1,custom|cmmlu:virology|0|1,custom|cmmlu:world_history|0|1,custom|cmmlu:world_religions|0|1</span>&quot;,&quot;<span class=\&quot;hljs-string\&quot;>===================</span>&quot;,&quot;<span class=\&quot;hljs-string\&quot;>custom|ceval:computer_network|0|1,custom|ceval:operating_system|0|1,custom|ceval:computer_architecture|0|1,custom|ceval:college_programming|0|1,custom|ceval:college_physics|0|1,custom|ceval:college_chemistry|0|1,custom|ceval:advanced_mathematics|0|1,custom|ceval:probability_and_statistics|0|1,custom|ceval:discrete_mathematics|0|1,custom|ceval:electrical_engineer|0|1,custom|ceval:metrology_engineer|0|1,custom|ceval:high_school_mathematics|0|1,custom|ceval:high_school_physics|0|1,custom|ceval:high_school_chemistry|0|1,custom|ceval:high_school_biology|0|1,custom|ceval:middle_school_mathematics|0|1,custom|ceval:middle_school_biology|0|1,custom|ceval:middle_school_physics|0|1,custom|ceval:middle_school_chemistry|0|1,custom|ceval:veterinary_medicine|0|1,custom|ceval:college_economics|0|1,custom|ceval:business_administration|0|1,custom|ceval:marxism|0|1,custom|ceval:mao_zedong_thought|0|1,custom|ceval:education_science|0|1,custom|ceval:teacher_qualification|0|1,custom|ceval:high_school_politics|0|1,custom|ceval:high_school_geography|0|1,custom|ceval:middle_school_politics|0|1,custom|ceval:middle_school_geography|0|1,custom|ceval:modern_chinese_history|0|1,custom|ceval:ideological_and_moral_cultivation|0|1,custom|ceval:logic|0|1,custom|ceval:law|0|1,custom|ceval:chinese_language_and_literature|0|1,custom|ceval:art_studies|0|1,custom|ceval:professional_tour_guide|0|1,custom|ceval:legal_professional|0|1,custom|ceval:high_school_chinese|0|1,custom|ceval:high_school_history|0|1,custom|ceval:middle_school_history|0|1,custom|ceval:civil_servant|0|1,custom|ceval:sports_science|0|1,custom|ceval:plant_protection|0|1,custom|ceval:basic_medicine|0|1,custom|ceval:clinical_medicine|0|1,custom|ceval:urban_and_rural_planner|0|1,custom|ceval:accountant|0|1,custom|ceval:fire_engineer|0|1,custom|ceval:environmental_impact_assessment_engineer|0|1,custom|ceval:tax_accountant|0|1,custom|ceval:physician|0|1</span>&quot;,&quot;<span class=\&quot;hljs-string\&quot;>===================</span>&quot;,&quot;<span class=\&quot;hljs-string\&quot;></span>&quot;,&quot;<span class=\&quot;hljs-string\&quot;>More info here: https://github.com/huggingface/lighteval?tab=readme-ov-file#evaluate-a-model-on-extended-community-or-custom-tasks</span>&quot;,&quot;<span class=\&quot;hljs-string\&quot;>For more info on differences between MMLU implementations: https://huggingface.co/blog/open-llm-leaderboard-mmlu#1001-flavors-of-mmlu</span>&quot;,&quot;<span class=\&quot;hljs-string\&quot;>In particular, the default leaderboard MMLU implementation (which uses &amp;quot;A&amp;quot;, &amp;quot;B&amp;quot;, etc as answer targets) gives generally random results on small/non instruction tuned models.</span>&quot;,&quot;<span class=\&quot;hljs-string\&quot;>Instead, we use the full MMLU answer as the target.</span>&quot;,&quot;<span class=\&quot;hljs-string\&quot;>&amp;quot;&amp;quot;&amp;quot;</span>&quot;,&quot;<span class=\&quot;hljs-keyword\&quot;>import</span> re&quot;,&quot;<span class=\&quot;hljs-keyword\&quot;>from</span> typing <span class=\&quot;hljs-keyword\&quot;>import</span> <span class=\&quot;hljs-type\&quot;>List</span>, <span class=\&quot;hljs-type\&quot;>Tuple</span>&quot;,&quot;&quot;,&quot;<span class=\&quot;hljs-keyword\&quot;>from</span> lighteval.metrics <span class=\&quot;hljs-keyword\&quot;>import</span> Metrics&quot;,&quot;<span class=\&quot;hljs-keyword\&quot;>from</span> lighteval.tasks.lighteval_task <span class=\&quot;hljs-keyword\&quot;>import</span> LightevalTaskConfig&quot;,&quot;<span class=\&quot;hljs-keyword\&quot;>from</span> lighteval.tasks.requests <span class=\&quot;hljs-keyword\&quot;>import</span> Doc&quot;,&quot;<span class=\&quot;hljs-keyword\&quot;>from</span> lighteval.tasks.tasks_prompt_formatting <span class=\&quot;hljs-keyword\&quot;>import</span> LETTER_INDICES&quot;,&quot;&quot;,&quot;_TASKS_STRINGS: <span class=\&quot;hljs-type\&quot;>List</span>[<span class=\&quot;hljs-type\&quot;>Tuple</span>[LightevalTaskConfig, <span class=\&quot;hljs-built_in\&quot;>str</span>]] = []&quot;,&quot;_TASKS: <span class=\&quot;hljs-type\&quot;>List</span>[LightevalTaskConfig] = []&quot;,&quot;&quot;,&quot;<span class=\&quot;hljs-comment\&quot;>## COMMON_SENSE_REASONING_TASKS ##</span>&quot;,&quot;COMMON_SENSE_REASONING_TASKS = [&quot;,&quot;    LightevalTaskConfig(&quot;,&quot;        name=<span class=\&quot;hljs-string\&quot;>&amp;quot;hellaswag&amp;quot;</span>,&quot;,&quot;        prompt_function=<span class=\&quot;hljs-string\&quot;>&amp;quot;hellaswag_prompt&amp;quot;</span>,&quot;,&quot;        hf_repo=<span class=\&quot;hljs-string\&quot;>&amp;quot;hellaswag&amp;quot;</span>,&quot;,&quot;        hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;default&amp;quot;</span>,&quot;,&quot;        metric=[<span class=\&quot;hljs-string\&quot;>&amp;quot;loglikelihood_acc&amp;quot;</span>, <span class=\&quot;hljs-string\&quot;>&amp;quot;loglikelihood_acc_norm_nospace&amp;quot;</span>],&quot;,&quot;    ),&quot;,&quot;    LightevalTaskConfig(&quot;,&quot;        name=<span class=\&quot;hljs-string\&quot;>&amp;quot;winogrande&amp;quot;</span>,&quot;,&quot;        prompt_function=<span class=\&quot;hljs-string\&quot;>&amp;quot;winogrande&amp;quot;</span>,&quot;,&quot;        hf_repo=<span class=\&quot;hljs-string\&quot;>&amp;quot;winogrande&amp;quot;</span>,&quot;,&quot;        hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;winogrande_xl&amp;quot;</span>,&quot;,&quot;        metric=[<span class=\&quot;hljs-string\&quot;>&amp;quot;loglikelihood_acc&amp;quot;</span>, <span class=\&quot;hljs-string\&quot;>&amp;quot;loglikelihood_acc_norm_nospace&amp;quot;</span>],&quot;,&quot;    ),&quot;,&quot;    LightevalTaskConfig(&quot;,&quot;        name=<span class=\&quot;hljs-string\&quot;>&amp;quot;piqa&amp;quot;</span>,&quot;,&quot;        prompt_function=<span class=\&quot;hljs-string\&quot;>&amp;quot;piqa_harness&amp;quot;</span>,&quot;,&quot;        hf_repo=<span class=\&quot;hljs-string\&quot;>&amp;quot;piqa&amp;quot;</span>,&quot;,&quot;        hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;plain_text&amp;quot;</span>,&quot;,&quot;        metric=[<span class=\&quot;hljs-string\&quot;>&amp;quot;loglikelihood_acc&amp;quot;</span>, <span class=\&quot;hljs-string\&quot;>&amp;quot;loglikelihood_acc_norm_nospace&amp;quot;</span>],&quot;,&quot;    ),&quot;,&quot;    LightevalTaskConfig(&quot;,&quot;        name=<span class=\&quot;hljs-string\&quot;>&amp;quot;siqa&amp;quot;</span>,&quot;,&quot;        prompt_function=<span class=\&quot;hljs-string\&quot;>&amp;quot;siqa_prompt&amp;quot;</span>,&quot;,&quot;        hf_repo=<span class=\&quot;hljs-string\&quot;>&amp;quot;lighteval/siqa&amp;quot;</span>,&quot;,&quot;        hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;default&amp;quot;</span>,&quot;,&quot;        hf_avail_splits=[<span class=\&quot;hljs-string\&quot;>&amp;quot;train&amp;quot;</span>, <span class=\&quot;hljs-string\&quot;>&amp;quot;validation&amp;quot;</span>],&quot;,&quot;        metric=[<span class=\&quot;hljs-string\&quot;>&amp;quot;loglikelihood_acc&amp;quot;</span>, <span class=\&quot;hljs-string\&quot;>&amp;quot;loglikelihood_acc_norm_nospace&amp;quot;</span>],&quot;,&quot;    ),&quot;,&quot;    LightevalTaskConfig(&quot;,&quot;        name=<span class=\&quot;hljs-string\&quot;>&amp;quot;openbookqa&amp;quot;</span>,&quot;,&quot;        prompt_function=<span class=\&quot;hljs-string\&quot;>&amp;quot;openbookqa&amp;quot;</span>,&quot;,&quot;        hf_repo=<span class=\&quot;hljs-string\&quot;>&amp;quot;openbookqa&amp;quot;</span>,&quot;,&quot;        hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;main&amp;quot;</span>,&quot;,&quot;        metric=[<span class=\&quot;hljs-string\&quot;>&amp;quot;loglikelihood_acc&amp;quot;</span>, <span class=\&quot;hljs-string\&quot;>&amp;quot;loglikelihood_acc_norm_nospace&amp;quot;</span>],&quot;,&quot;    ),&quot;,&quot;    LightevalTaskConfig(&quot;,&quot;        name=<span class=\&quot;hljs-string\&quot;>&amp;quot;arc:easy&amp;quot;</span>,&quot;,&quot;        prompt_function=<span class=\&quot;hljs-string\&quot;>&amp;quot;arc&amp;quot;</span>,&quot;,&quot;        hf_repo=<span class=\&quot;hljs-string\&quot;>&amp;quot;ai2_arc&amp;quot;</span>,&quot;,&quot;        hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;ARC-Easy&amp;quot;</span>,&quot;,&quot;        evaluation_splits=[<span class=\&quot;hljs-string\&quot;>&amp;quot;test&amp;quot;</span>],&quot;,&quot;        generation_size=<span class=\&quot;hljs-number\&quot;>1</span>,&quot;,&quot;        metric=[<span class=\&quot;hljs-string\&quot;>&amp;quot;loglikelihood_acc&amp;quot;</span>, <span class=\&quot;hljs-string\&quot;>&amp;quot;loglikelihood_acc_norm_nospace&amp;quot;</span>],&quot;,&quot;    ),&quot;,&quot;    LightevalTaskConfig(&quot;,&quot;        name=<span class=\&quot;hljs-string\&quot;>&amp;quot;arc:challenge&amp;quot;</span>,&quot;,&quot;        prompt_function=<span class=\&quot;hljs-string\&quot;>&amp;quot;arc&amp;quot;</span>,&quot;,&quot;        hf_repo=<span class=\&quot;hljs-string\&quot;>&amp;quot;ai2_arc&amp;quot;</span>,&quot;,&quot;        hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;ARC-Challenge&amp;quot;</span>,&quot;,&quot;        evaluation_splits=[<span class=\&quot;hljs-string\&quot;>&amp;quot;test&amp;quot;</span>],&quot;,&quot;        generation_size=<span class=\&quot;hljs-number\&quot;>1</span>,&quot;,&quot;        metric=[<span class=\&quot;hljs-string\&quot;>&amp;quot;loglikelihood_acc&amp;quot;</span>, <span class=\&quot;hljs-string\&quot;>&amp;quot;loglikelihood_acc_norm_nospace&amp;quot;</span>],&quot;,&quot;    ),&quot;,&quot;    LightevalTaskConfig(&quot;,&quot;        name=<span class=\&quot;hljs-string\&quot;>&amp;quot;commonsense_qa&amp;quot;</span>,&quot;,&quot;        prompt_function=<span class=\&quot;hljs-string\&quot;>&amp;quot;commonsense_qa_prompt&amp;quot;</span>,&quot;,&quot;        hf_repo=<span class=\&quot;hljs-string\&quot;>&amp;quot;commonsense_qa&amp;quot;</span>,&quot;,&quot;        hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;default&amp;quot;</span>,&quot;,&quot;        metric=[<span class=\&quot;hljs-string\&quot;>&amp;quot;loglikelihood_acc&amp;quot;</span>, <span class=\&quot;hljs-string\&quot;>&amp;quot;loglikelihood_acc_norm_nospace&amp;quot;</span>],&quot;,&quot;    ),&quot;,&quot;]&quot;,&quot;&quot;,&quot;&quot;,&quot;<span class=\&quot;hljs-keyword\&quot;>def</span> <span class=\&quot;hljs-title function_\&quot;>commonsense_qa_prompt</span>(<span class=\&quot;hljs-params\&quot;>line, task_name: <span class=\&quot;hljs-built_in\&quot;>str</span> = <span class=\&quot;hljs-literal\&quot;>None</span></span>):&quot;,&quot;    <span class=\&quot;hljs-keyword\&quot;>return</span> Doc(&quot;,&quot;        task_name=task_name,&quot;,&quot;        query=line[<span class=\&quot;hljs-string\&quot;>&amp;quot;question&amp;quot;</span>],&quot;,&quot;        choices=[<span class=\&quot;hljs-string\&quot;>f&amp;quot; <span class=\&quot;hljs-subst\&quot;>{c}</span>&amp;quot;</span> <span class=\&quot;hljs-keyword\&quot;>for</span> c <span class=\&quot;hljs-keyword\&quot;>in</span> line[<span class=\&quot;hljs-string\&quot;>&amp;quot;choices&amp;quot;</span>][<span class=\&quot;hljs-string\&quot;>&amp;quot;text&amp;quot;</span>]],&quot;,&quot;        gold_index=LETTER_INDICES.index(line[<span class=\&quot;hljs-string\&quot;>&amp;quot;answerKey&amp;quot;</span>].strip()),&quot;,&quot;        instruction=<span class=\&quot;hljs-string\&quot;>&amp;quot;&amp;quot;</span>,&quot;,&quot;    )&quot;,&quot;&quot;,&quot;&quot;,&quot;<span class=\&quot;hljs-keyword\&quot;>def</span> <span class=\&quot;hljs-title function_\&quot;>siqa_prompt</span>(<span class=\&quot;hljs-params\&quot;>line, task_name: <span class=\&quot;hljs-built_in\&quot;>str</span> = <span class=\&quot;hljs-literal\&quot;>None</span></span>):&quot;,&quot;    <span class=\&quot;hljs-keyword\&quot;>return</span> Doc(&quot;,&quot;        task_name=task_name,&quot;,&quot;        query=line[<span class=\&quot;hljs-string\&quot;>&amp;quot;context&amp;quot;</span>] + <span class=\&quot;hljs-string\&quot;>&amp;quot; &amp;quot;</span> + line[<span class=\&quot;hljs-string\&quot;>&amp;quot;question&amp;quot;</span>],&quot;,&quot;        choices=[<span class=\&quot;hljs-string\&quot;>f&amp;quot; <span class=\&quot;hljs-subst\&quot;>{c}</span>&amp;quot;</span> <span class=\&quot;hljs-keyword\&quot;>for</span> c <span class=\&quot;hljs-keyword\&quot;>in</span> [line[<span class=\&quot;hljs-string\&quot;>&amp;quot;answerA&amp;quot;</span>], line[<span class=\&quot;hljs-string\&quot;>&amp;quot;answerB&amp;quot;</span>], line[<span class=\&quot;hljs-string\&quot;>&amp;quot;answerC&amp;quot;</span>]]],&quot;,&quot;        gold_index=<span class=\&quot;hljs-built_in\&quot;>int</span>(line[<span class=\&quot;hljs-string\&quot;>&amp;quot;label&amp;quot;</span>]) - <span class=\&quot;hljs-number\&quot;>1</span>,&quot;,&quot;        instruction=<span class=\&quot;hljs-string\&quot;>&amp;quot;&amp;quot;</span>,&quot;,&quot;    )&quot;,&quot;&quot;,&quot;&quot;,&quot;<span class=\&quot;hljs-keyword\&quot;>def</span> <span class=\&quot;hljs-title function_\&quot;>hellaswag_prompt</span>(<span class=\&quot;hljs-params\&quot;>line, task_name: <span class=\&quot;hljs-built_in\&quot;>str</span> = <span class=\&quot;hljs-literal\&quot;>None</span></span>):&quot;,&quot;    <span class=\&quot;hljs-keyword\&quot;>def</span> <span class=\&quot;hljs-title function_\&quot;>preprocess</span>(<span class=\&quot;hljs-params\&quot;>text</span>):&quot;,&quot;        <span class=\&quot;hljs-string\&quot;>&amp;quot;&amp;quot;&amp;quot;Comes from AiHarness&amp;quot;&amp;quot;&amp;quot;</span>&quot;,&quot;        <span class=\&quot;hljs-comment\&quot;># text = text.strip()</span>&quot;,&quot;        <span class=\&quot;hljs-comment\&quot;># <span class=\&quot;hljs-doctag\&quot;>NOTE:</span> Brackets are artifacts of the WikiHow dataset portion of HellaSwag.</span>&quot;,&quot;        text = text.replace(<span class=\&quot;hljs-string\&quot;>&amp;quot; [title]&amp;quot;</span>, <span class=\&quot;hljs-string\&quot;>&amp;quot;. &amp;quot;</span>)&quot;,&quot;        text = re.sub(<span class=\&quot;hljs-string\&quot;>&amp;quot;\\\\[.*?\\\\]&amp;quot;</span>, <span class=\&quot;hljs-string\&quot;>&amp;quot;&amp;quot;</span>, text)&quot;,&quot;        text = text.replace(<span class=\&quot;hljs-string\&quot;>&amp;quot;  &amp;quot;</span>, <span class=\&quot;hljs-string\&quot;>&amp;quot; &amp;quot;</span>)&quot;,&quot;        <span class=\&quot;hljs-keyword\&quot;>return</span> text&quot;,&quot;&quot;,&quot;    ctx = <span class=\&quot;hljs-string\&quot;>f&amp;quot;<span class=\&quot;hljs-subst\&quot;>{line[<span class=\&quot;hljs-string\&quot;>&amp;#x27;ctx_a&amp;#x27;</span>]}</span> <span class=\&quot;hljs-subst\&quot;>{line[<span class=\&quot;hljs-string\&quot;>&amp;#x27;ctx_b&amp;#x27;</span>].capitalize()}</span> &amp;quot;</span>&quot;,&quot;    <span class=\&quot;hljs-keyword\&quot;>return</span> Doc(&quot;,&quot;        task_name=task_name,&quot;,&quot;        query=preprocess(line[<span class=\&quot;hljs-string\&quot;>&amp;quot;activity_label&amp;quot;</span>] + <span class=\&quot;hljs-string\&quot;>&amp;quot;: &amp;quot;</span> + ctx),&quot;,&quot;        choices=[<span class=\&quot;hljs-string\&quot;>&amp;quot; &amp;quot;</span> + preprocess(ending) <span class=\&quot;hljs-keyword\&quot;>for</span> ending <span class=\&quot;hljs-keyword\&quot;>in</span> line[<span class=\&quot;hljs-string\&quot;>&amp;quot;endings&amp;quot;</span>]],&quot;,&quot;        gold_index=<span class=\&quot;hljs-built_in\&quot;>int</span>(line[<span class=\&quot;hljs-string\&quot;>&amp;quot;label&amp;quot;</span>]) <span class=\&quot;hljs-keyword\&quot;>if</span> line[<span class=\&quot;hljs-string\&quot;>&amp;quot;label&amp;quot;</span>] != <span class=\&quot;hljs-string\&quot;>&amp;quot;&amp;quot;</span> <span class=\&quot;hljs-keyword\&quot;>else</span> -<span class=\&quot;hljs-number\&quot;>1</span>,  <span class=\&quot;hljs-comment\&quot;># -1 for test</span>&quot;,&quot;        <span class=\&quot;hljs-comment\&quot;># &amp;quot;metric&amp;quot;: &amp;quot;choices_loglikelihood&amp;quot;,</span>&quot;,&quot;    )&quot;,&quot;&quot;,&quot;&quot;,&quot;<span class=\&quot;hljs-comment\&quot;># 0 short for common sense</span>&quot;,&quot;COMMON_SENSE_REASONING_STRING = [(t, <span class=\&quot;hljs-string\&quot;>f&amp;quot;custom|<span class=\&quot;hljs-subst\&quot;>{t.name}</span>|0|1&amp;quot;</span>) <span class=\&quot;hljs-keyword\&quot;>for</span> t <span class=\&quot;hljs-keyword\&quot;>in</span> COMMON_SENSE_REASONING_TASKS]&quot;,&quot;_TASKS_STRINGS.extend(COMMON_SENSE_REASONING_STRING)&quot;,&quot;_TASKS += COMMON_SENSE_REASONING_TASKS&quot;,&quot;&quot;,&quot;<span class=\&quot;hljs-comment\&quot;>## MMLU ##</span>&quot;,&quot;<span class=\&quot;hljs-keyword\&quot;>class</span> <span class=\&quot;hljs-title class_\&quot;>CustomMMLUEvaluationTask</span>(<span class=\&quot;hljs-title class_ inherited__\&quot;>LightevalTaskConfig</span>):&quot;,&quot;    <span class=\&quot;hljs-keyword\&quot;>def</span> <span class=\&quot;hljs-title function_\&quot;>__init__</span>(<span class=\&quot;hljs-params\&quot;></span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        self,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        name,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        prompt_function=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu_prompt&amp;quot;</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        hf_repo=<span class=\&quot;hljs-string\&quot;>&amp;quot;lighteval/mmlu&amp;quot;</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        hf_subset=<span class=\&quot;hljs-literal\&quot;>None</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        <span class=\&quot;hljs-comment\&quot;>#  metric=[Metrics.loglikelihood_acc_single_token],</span></span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        hf_avail_splits=<span class=\&quot;hljs-literal\&quot;>None</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        evaluation_splits=[<span class=\&quot;hljs-string\&quot;>&amp;quot;test&amp;quot;</span>],</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        few_shots_split=<span class=\&quot;hljs-string\&quot;>&amp;quot;dev&amp;quot;</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        few_shots_select=<span class=\&quot;hljs-literal\&quot;>None</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        suite=<span class=\&quot;hljs-literal\&quot;>None</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        generation_size=-<span class=\&quot;hljs-number\&quot;>1</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        stop_sequence=<span class=\&quot;hljs-literal\&quot;>None</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        output_regex=<span class=\&quot;hljs-literal\&quot;>None</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        frozen=<span class=\&quot;hljs-literal\&quot;>False</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>    </span>):&quot;,&quot;        <span class=\&quot;hljs-built_in\&quot;>super</span>().__init__(&quot;,&quot;            name=name,&quot;,&quot;            prompt_function=prompt_function,&quot;,&quot;            hf_repo=hf_repo,&quot;,&quot;            hf_subset=hf_subset,&quot;,&quot;            metric=metric,&quot;,&quot;            hf_avail_splits=hf_avail_splits,&quot;,&quot;            evaluation_splits=evaluation_splits,&quot;,&quot;            few_shots_split=few_shots_split,&quot;,&quot;            few_shots_select=few_shots_select,&quot;,&quot;            suite=suite,&quot;,&quot;            generation_size=generation_size,&quot;,&quot;            stop_sequence=stop_sequence,&quot;,&quot;            output_regex=output_regex,&quot;,&quot;            frozen=frozen,&quot;,&quot;        )&quot;,&quot;&quot;,&quot;&quot;,&quot;MMLU_TASKS = [&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:abstract_algebra&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;abstract_algebra&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:anatomy&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;anatomy&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:astronomy&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;astronomy&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:business_ethics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;business_ethics&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:clinical_knowledge&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;clinical_knowledge&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:college_biology&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;college_biology&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:college_chemistry&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;college_chemistry&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:college_computer_science&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;college_computer_science&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:college_mathematics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;college_mathematics&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:college_medicine&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;college_medicine&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:college_physics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;college_physics&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:computer_security&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;computer_security&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:conceptual_physics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;conceptual_physics&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:econometrics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;econometrics&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:electrical_engineering&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;electrical_engineering&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:elementary_mathematics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;elementary_mathematics&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:formal_logic&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;formal_logic&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:global_facts&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;global_facts&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:high_school_biology&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_biology&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:high_school_chemistry&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_chemistry&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:high_school_computer_science&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_computer_science&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:high_school_european_history&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_european_history&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:high_school_geography&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_geography&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(&quot;,&quot;        name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:high_school_government_and_politics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_government_and_politics&amp;quot;</span>&quot;,&quot;    ),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:high_school_macroeconomics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_macroeconomics&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:high_school_mathematics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_mathematics&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:high_school_microeconomics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_microeconomics&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:high_school_physics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_physics&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:high_school_psychology&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_psychology&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:high_school_statistics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_statistics&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:high_school_us_history&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_us_history&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:high_school_world_history&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_world_history&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:human_aging&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;human_aging&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:human_sexuality&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;human_sexuality&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:international_law&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;international_law&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:jurisprudence&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;jurisprudence&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:logical_fallacies&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;logical_fallacies&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:machine_learning&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;machine_learning&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:management&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;management&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:marketing&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;marketing&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:medical_genetics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;medical_genetics&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:miscellaneous&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;miscellaneous&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:moral_disputes&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;moral_disputes&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:moral_scenarios&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;moral_scenarios&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:nutrition&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;nutrition&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:philosophy&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;philosophy&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:prehistory&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;prehistory&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:professional_accounting&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;professional_accounting&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:professional_law&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;professional_law&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:professional_medicine&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;professional_medicine&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:professional_psychology&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;professional_psychology&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:public_relations&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;public_relations&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:security_studies&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;security_studies&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:sociology&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;sociology&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:us_foreign_policy&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;us_foreign_policy&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:virology&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;virology&amp;quot;</span>),&quot;,&quot;    CustomMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;mmlu:world_religions&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;world_religions&amp;quot;</span>),&quot;,&quot;]&quot;,&quot;&quot;,&quot;&quot;,&quot;<span class=\&quot;hljs-keyword\&quot;>def</span> <span class=\&quot;hljs-title function_\&quot;>mmlu_prompt</span>(<span class=\&quot;hljs-params\&quot;>line, task_name: <span class=\&quot;hljs-built_in\&quot;>str</span> = <span class=\&quot;hljs-literal\&quot;>None</span></span>):&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;quot;&amp;quot;&amp;quot;MMLU prompt without letters&amp;quot;&amp;quot;&amp;quot;</span>&quot;,&quot;    topic = line[<span class=\&quot;hljs-string\&quot;>&amp;quot;subject&amp;quot;</span>]&quot;,&quot;    prompt = <span class=\&quot;hljs-string\&quot;>f&amp;quot;The following are questions about <span class=\&quot;hljs-subst\&quot;>{topic.replace(<span class=\&quot;hljs-string\&quot;>&amp;#x27;_&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27; &amp;#x27;</span>)}</span>.\\nQuestion: &amp;quot;</span>&quot;,&quot;    prompt += line[<span class=\&quot;hljs-string\&quot;>&amp;quot;question&amp;quot;</span>] + <span class=\&quot;hljs-string\&quot;>&amp;quot;\\nAnswer:&amp;quot;</span>&quot;,&quot;    <span class=\&quot;hljs-comment\&quot;>#print(f&amp;quot;mmlu_prompt={prompt}&amp;quot;)</span>&quot;,&quot;&quot;,&quot;    <span class=\&quot;hljs-keyword\&quot;>return</span> Doc(&quot;,&quot;        task_name=task_name,&quot;,&quot;        query=prompt,&quot;,&quot;        choices=[<span class=\&quot;hljs-string\&quot;>f&amp;quot; <span class=\&quot;hljs-subst\&quot;>{c}</span>&amp;quot;</span> <span class=\&quot;hljs-keyword\&quot;>for</span> c <span class=\&quot;hljs-keyword\&quot;>in</span> line[<span class=\&quot;hljs-string\&quot;>&amp;quot;choices&amp;quot;</span>]],&quot;,&quot;        gold_index=line[<span class=\&quot;hljs-string\&quot;>&amp;quot;answer&amp;quot;</span>],&quot;,&quot;        instruction=<span class=\&quot;hljs-string\&quot;>f&amp;quot;The following are questions about <span class=\&quot;hljs-subst\&quot;>{topic.replace(<span class=\&quot;hljs-string\&quot;>&amp;#x27;_&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27; &amp;#x27;</span>)}</span>.\\n&amp;quot;</span>,&quot;,&quot;    )&quot;,&quot;&quot;,&quot;&quot;,&quot;MMLU_STRING = [(t, <span class=\&quot;hljs-string\&quot;>f&amp;quot;custom|<span class=\&quot;hljs-subst\&quot;>{t.name}</span>|0|1&amp;quot;</span>) <span class=\&quot;hljs-keyword\&quot;>for</span> t <span class=\&quot;hljs-keyword\&quot;>in</span> MMLU_TASKS]&quot;,&quot;_TASKS_STRINGS.extend(MMLU_STRING)&quot;,&quot;_TASKS += MMLU_TASKS&quot;,&quot;&quot;,&quot;&quot;,&quot;<span class=\&quot;hljs-comment\&quot;>############################################################################################################################################################</span>&quot;,&quot;<span class=\&quot;hljs-comment\&quot;>## CMMLU ##</span>&quot;,&quot;<span class=\&quot;hljs-keyword\&quot;>class</span> <span class=\&quot;hljs-title class_\&quot;>CustomCMMLUEvaluationTask</span>(<span class=\&quot;hljs-title class_ inherited__\&quot;>LightevalTaskConfig</span>):&quot;,&quot;    <span class=\&quot;hljs-keyword\&quot;>def</span> <span class=\&quot;hljs-title function_\&quot;>__init__</span>(<span class=\&quot;hljs-params\&quot;></span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        self,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        name,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        prompt_function=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu_prompt&amp;quot;</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        hf_repo=<span class=\&quot;hljs-string\&quot;>&amp;quot;ldwang/lighteval-cmmlu&amp;quot;</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        hf_subset=<span class=\&quot;hljs-literal\&quot;>None</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        <span class=\&quot;hljs-comment\&quot;>#  metric=[Metrics.loglikelihood_acc_single_token],</span></span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        hf_avail_splits=<span class=\&quot;hljs-literal\&quot;>None</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        evaluation_splits=[<span class=\&quot;hljs-string\&quot;>&amp;quot;test&amp;quot;</span>],</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        few_shots_split=<span class=\&quot;hljs-string\&quot;>&amp;quot;dev&amp;quot;</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        few_shots_select=<span class=\&quot;hljs-literal\&quot;>None</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        suite=<span class=\&quot;hljs-literal\&quot;>None</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        generation_size=-<span class=\&quot;hljs-number\&quot;>1</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        stop_sequence=<span class=\&quot;hljs-literal\&quot;>None</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        output_regex=<span class=\&quot;hljs-literal\&quot;>None</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        frozen=<span class=\&quot;hljs-literal\&quot;>False</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>    </span>):&quot;,&quot;        <span class=\&quot;hljs-built_in\&quot;>super</span>().__init__(&quot;,&quot;            name=name,&quot;,&quot;            prompt_function=prompt_function,&quot;,&quot;            hf_repo=hf_repo,&quot;,&quot;            hf_subset=hf_subset,&quot;,&quot;            metric=metric,&quot;,&quot;            hf_avail_splits=hf_avail_splits,&quot;,&quot;            evaluation_splits=evaluation_splits,&quot;,&quot;            few_shots_split=few_shots_split,&quot;,&quot;            few_shots_select=few_shots_select,&quot;,&quot;            suite=suite,&quot;,&quot;            generation_size=generation_size,&quot;,&quot;            stop_sequence=stop_sequence,&quot;,&quot;            output_regex=output_regex,&quot;,&quot;            frozen=frozen,&quot;,&quot;            trust_dataset=<span class=\&quot;hljs-literal\&quot;>True</span>,&quot;,&quot;        )&quot;,&quot;&quot;,&quot;&quot;,&quot;CMMLU_TASKS = [&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:agronomy&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;agronomy&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:anatomy&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;anatomy&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:ancient_chinese&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;ancient_chinese&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:arts&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;arts&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:astronomy&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;astronomy&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:business_ethics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;business_ethics&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:chinese_civil_service_exam&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;chinese_civil_service_exam&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:chinese_driving_rule&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;chinese_driving_rule&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:chinese_food_culture&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;chinese_food_culture&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:chinese_foreign_policy&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;chinese_foreign_policy&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:chinese_history&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;chinese_history&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:chinese_literature&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;chinese_literature&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:chinese_teacher_qualification&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;chinese_teacher_qualification&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:clinical_knowledge&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;clinical_knowledge&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:college_actuarial_science&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;college_actuarial_science&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:college_education&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;college_education&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:college_engineering_hydrology&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;college_engineering_hydrology&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:college_law&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;college_law&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:college_mathematics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;college_mathematics&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:college_medical_statistics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;college_medical_statistics&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:college_medicine&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;college_medicine&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:computer_science&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;computer_science&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:computer_security&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;computer_security&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:conceptual_physics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;conceptual_physics&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:construction_project_management&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;construction_project_management&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:economics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;economics&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:education&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;education&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:electrical_engineering&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;electrical_engineering&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:elementary_chinese&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;elementary_chinese&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:elementary_commonsense&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;elementary_commonsense&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:elementary_information_and_technology&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;elementary_information_and_technology&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:elementary_mathematics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;elementary_mathematics&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:ethnology&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;ethnology&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:food_science&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;food_science&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:genetics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;genetics&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:global_facts&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;global_facts&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:high_school_biology&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_biology&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:high_school_chemistry&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_chemistry&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:high_school_geography&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_geography&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:high_school_mathematics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_mathematics&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:high_school_physics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_physics&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:high_school_politics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_politics&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:human_sexuality&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;human_sexuality&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:international_law&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;international_law&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:journalism&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;journalism&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:jurisprudence&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;jurisprudence&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:legal_and_moral_basis&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;legal_and_moral_basis&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:logical&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;logical&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:machine_learning&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;machine_learning&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:management&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;management&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:marketing&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;marketing&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:marxist_theory&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;marxist_theory&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:modern_chinese&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;modern_chinese&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:nutrition&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;nutrition&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:philosophy&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;philosophy&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:professional_accounting&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;professional_accounting&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:professional_law&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;professional_law&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:professional_medicine&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;professional_medicine&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:professional_psychology&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;professional_psychology&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:public_relations&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;public_relations&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:security_study&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;security_study&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:sociology&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;sociology&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:sports_science&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;sports_science&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:traditional_chinese_medicine&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;traditional_chinese_medicine&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:virology&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;virology&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:world_history&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;world_history&amp;quot;</span>),&quot;,&quot;\t\tCustomCMMLUEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;cmmlu:world_religions&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;world_religions&amp;quot;</span>),&quot;,&quot;]&quot;,&quot;&quot;,&quot;cmmlu_subject_mapping = {&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;agronomy&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;农学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;anatomy&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;解剖学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;ancient_chinese&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;古汉语&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;arts&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;艺术学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;astronomy&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;天文学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;business_ethics&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;商业伦理&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;chinese_civil_service_exam&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;中国公务员考试&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;chinese_driving_rule&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;中国驾驶规则&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;chinese_food_culture&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;中国饮食文化&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;chinese_foreign_policy&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;中国外交政策&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;chinese_history&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;中国历史&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;chinese_literature&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;中国文学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;chinese_teacher_qualification&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;中国教师资格&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;clinical_knowledge&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;临床知识&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;college_actuarial_science&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;大学精算学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;college_education&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;大学教育学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;college_engineering_hydrology&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;大学工程水文学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;college_law&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;大学法律&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;college_mathematics&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;大学数学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;college_medical_statistics&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;大学医学统计&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;college_medicine&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;大学医学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;computer_science&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;计算机科学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;computer_security&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;计算机安全&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;conceptual_physics&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;概念物理学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;construction_project_management&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;建设工程管理&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;economics&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;经济学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;education&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;教育学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;electrical_engineering&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;电气工程&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;elementary_chinese&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;小学语文&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;elementary_commonsense&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;小学常识&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;elementary_information_and_technology&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;小学信息技术&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;elementary_mathematics&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;初等数学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;ethnology&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;民族学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;food_science&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;食品科学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;genetics&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;遗传学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;global_facts&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;全球事实&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;high_school_biology&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;高中生物&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;high_school_chemistry&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;高中化学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;high_school_geography&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;高中地理&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;high_school_mathematics&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;高中数学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;high_school_physics&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;高中物理学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;high_school_politics&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;高中政治&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;human_sexuality&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;人类性行为&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;international_law&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;国际法学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;journalism&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;新闻学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;jurisprudence&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;法理学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;legal_and_moral_basis&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;法律与道德基础&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;logical&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;逻辑学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;machine_learning&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;机器学习&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;management&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;管理学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;marketing&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;市场营销&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;marxist_theory&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;马克思主义理论&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;modern_chinese&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;现代汉语&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;nutrition&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;营养学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;philosophy&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;哲学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;professional_accounting&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;专业会计&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;professional_law&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;专业法学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;professional_medicine&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;专业医学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;professional_psychology&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;专业心理学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;public_relations&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;公共关系&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;security_study&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;安全研究&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;sociology&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;社会学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;sports_science&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;体育学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;traditional_chinese_medicine&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;中医中药&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;virology&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;病毒学&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;world_history&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;世界历史&amp;#x27;</span>,&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;world_religions&amp;#x27;</span>: <span class=\&quot;hljs-string\&quot;>&amp;#x27;世界宗教&amp;#x27;</span>&quot;,&quot;}&quot;,&quot;&quot;,&quot;<span class=\&quot;hljs-keyword\&quot;>def</span> <span class=\&quot;hljs-title function_\&quot;>cmmlu_prompt</span>(<span class=\&quot;hljs-params\&quot;>line, task_name: <span class=\&quot;hljs-built_in\&quot;>str</span> = <span class=\&quot;hljs-literal\&quot;>None</span></span>):&quot;,&quot;    <span class=\&quot;hljs-comment\&quot;># 以下是关于{_ch_name}的单项选择题，请直接给出正确答案的选项。\\n题目：{{question}}\\nA. {{A}}\\nB. {{B}}\\nC. {{C}}\\nD. {{D}}</span>&quot;,&quot;    <span class=\&quot;hljs-comment\&quot;># 答案是: {{{answer}}}</span>&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;quot;&amp;quot;&amp;quot;CMMLU prompt without letters&amp;quot;&amp;quot;&amp;quot;</span>&quot;,&quot;    topic = cmmlu_subject_mapping[line[<span class=\&quot;hljs-string\&quot;>&amp;#x27;subject&amp;#x27;</span>]]&quot;,&quot;    prompt = <span class=\&quot;hljs-string\&quot;>f&amp;quot;以下是关于<span class=\&quot;hljs-subst\&quot;>{topic.replace(<span class=\&quot;hljs-string\&quot;>&amp;#x27;_&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27; &amp;#x27;</span>)}</span>的单项选择题，请直接给出正确答案的选项。\\n题目：&amp;quot;</span>&quot;,&quot;    prompt += line[<span class=\&quot;hljs-string\&quot;>&amp;quot;question&amp;quot;</span>] + <span class=\&quot;hljs-string\&quot;>&amp;quot;\\n答案是：&amp;quot;</span>&quot;,&quot;    <span class=\&quot;hljs-comment\&quot;>#print(f&amp;quot;cmmlu_prompt={prompt}&amp;quot;)</span>&quot;,&quot;&quot;,&quot;    <span class=\&quot;hljs-keyword\&quot;>return</span> Doc(&quot;,&quot;        task_name=task_name,&quot;,&quot;        query=prompt,&quot;,&quot;        choices=[<span class=\&quot;hljs-string\&quot;>f&amp;quot; <span class=\&quot;hljs-subst\&quot;>{c}</span>&amp;quot;</span> <span class=\&quot;hljs-keyword\&quot;>for</span> c <span class=\&quot;hljs-keyword\&quot;>in</span> line[<span class=\&quot;hljs-string\&quot;>&amp;quot;choices&amp;quot;</span>]],&quot;,&quot;        gold_index=line[<span class=\&quot;hljs-string\&quot;>&amp;quot;answer&amp;quot;</span>],&quot;,&quot;        instruction=<span class=\&quot;hljs-literal\&quot;>None</span>,&quot;,&quot;    )&quot;,&quot;&quot;,&quot;CMMLU_STRING = [(t, <span class=\&quot;hljs-string\&quot;>f&amp;quot;custom|<span class=\&quot;hljs-subst\&quot;>{t.name}</span>|0|1&amp;quot;</span>) <span class=\&quot;hljs-keyword\&quot;>for</span> t <span class=\&quot;hljs-keyword\&quot;>in</span> CMMLU_TASKS]&quot;,&quot;_TASKS_STRINGS.extend(CMMLU_STRING)&quot;,&quot;_TASKS += CMMLU_TASKS&quot;,&quot;<span class=\&quot;hljs-built_in\&quot;>print</span>(<span class=\&quot;hljs-string\&quot;>f&amp;#x27;<span class=\&quot;hljs-subst\&quot;>{<span class=\&quot;hljs-string\&quot;>&amp;quot;,&amp;quot;</span>.join([t[<span class=\&quot;hljs-number\&quot;>1</span>] <span class=\&quot;hljs-keyword\&quot;>for</span> t <span class=\&quot;hljs-keyword\&quot;>in</span> CMMLU_STRING])}</span>&amp;#x27;</span>)&quot;,&quot;&quot;,&quot;<span class=\&quot;hljs-comment\&quot;>############################################################################################################################################################</span>&quot;,&quot;<span class=\&quot;hljs-comment\&quot;>## CEVAL ##</span>&quot;,&quot;<span class=\&quot;hljs-keyword\&quot;>class</span> <span class=\&quot;hljs-title class_\&quot;>CustomCEVALEvaluationTask</span>(<span class=\&quot;hljs-title class_ inherited__\&quot;>LightevalTaskConfig</span>):&quot;,&quot;    <span class=\&quot;hljs-keyword\&quot;>def</span> <span class=\&quot;hljs-title function_\&quot;>__init__</span>(<span class=\&quot;hljs-params\&quot;></span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        self,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        name,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        prompt_function=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval_prompt&amp;quot;</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        hf_repo=<span class=\&quot;hljs-string\&quot;>&amp;quot;ldwang/lighteval-ceval-exam&amp;quot;</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        hf_subset=<span class=\&quot;hljs-literal\&quot;>None</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        <span class=\&quot;hljs-comment\&quot;>#  metric=[Metrics.loglikelihood_acc_single_token],</span></span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        hf_avail_splits=<span class=\&quot;hljs-literal\&quot;>None</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        evaluation_splits=[<span class=\&quot;hljs-string\&quot;>&amp;quot;val&amp;quot;</span>],</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        few_shots_split=<span class=\&quot;hljs-string\&quot;>&amp;quot;dev&amp;quot;</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        few_shots_select=<span class=\&quot;hljs-literal\&quot;>None</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        suite=<span class=\&quot;hljs-literal\&quot;>None</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        generation_size=-<span class=\&quot;hljs-number\&quot;>1</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        stop_sequence=<span class=\&quot;hljs-literal\&quot;>None</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        output_regex=<span class=\&quot;hljs-literal\&quot;>None</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>        frozen=<span class=\&quot;hljs-literal\&quot;>False</span>,</span>&quot;,&quot;<span class=\&quot;hljs-params\&quot;>    </span>):&quot;,&quot;        <span class=\&quot;hljs-built_in\&quot;>super</span>().__init__(&quot;,&quot;            name=name,&quot;,&quot;            prompt_function=prompt_function,&quot;,&quot;            hf_repo=hf_repo,&quot;,&quot;            hf_subset=hf_subset,&quot;,&quot;            metric=metric,&quot;,&quot;            hf_avail_splits=hf_avail_splits,&quot;,&quot;            evaluation_splits=evaluation_splits,&quot;,&quot;            few_shots_split=few_shots_split,&quot;,&quot;            few_shots_select=few_shots_select,&quot;,&quot;            suite=suite,&quot;,&quot;            generation_size=generation_size,&quot;,&quot;            stop_sequence=stop_sequence,&quot;,&quot;            output_regex=output_regex,&quot;,&quot;            frozen=frozen,&quot;,&quot;            trust_dataset=<span class=\&quot;hljs-literal\&quot;>True</span>,&quot;,&quot;        )&quot;,&quot;&quot;,&quot;&quot;,&quot;CEVAL_TASKS = [&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:computer_network&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;computer_network&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:operating_system&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;operating_system&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:computer_architecture&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;computer_architecture&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:college_programming&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;college_programming&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:college_physics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;college_physics&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:college_chemistry&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;college_chemistry&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:advanced_mathematics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;advanced_mathematics&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:probability_and_statistics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;probability_and_statistics&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:discrete_mathematics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;discrete_mathematics&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:electrical_engineer&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;electrical_engineer&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:metrology_engineer&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;metrology_engineer&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:high_school_mathematics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_mathematics&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:high_school_physics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_physics&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:high_school_chemistry&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_chemistry&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:high_school_biology&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_biology&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:middle_school_mathematics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;middle_school_mathematics&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:middle_school_biology&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;middle_school_biology&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:middle_school_physics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;middle_school_physics&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:middle_school_chemistry&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;middle_school_chemistry&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:veterinary_medicine&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;veterinary_medicine&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:college_economics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;college_economics&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:business_administration&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;business_administration&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:marxism&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;marxism&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:mao_zedong_thought&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;mao_zedong_thought&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:education_science&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;education_science&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:teacher_qualification&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;teacher_qualification&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:high_school_politics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_politics&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:high_school_geography&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_geography&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:middle_school_politics&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;middle_school_politics&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:middle_school_geography&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;middle_school_geography&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:modern_chinese_history&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;modern_chinese_history&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:ideological_and_moral_cultivation&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;ideological_and_moral_cultivation&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:logic&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;logic&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:law&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;law&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:chinese_language_and_literature&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;chinese_language_and_literature&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:art_studies&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;art_studies&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:professional_tour_guide&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;professional_tour_guide&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:legal_professional&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;legal_professional&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:high_school_chinese&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_chinese&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:high_school_history&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;high_school_history&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:middle_school_history&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;middle_school_history&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:civil_servant&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;civil_servant&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:sports_science&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;sports_science&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:plant_protection&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;plant_protection&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:basic_medicine&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;basic_medicine&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:clinical_medicine&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;clinical_medicine&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:urban_and_rural_planner&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;urban_and_rural_planner&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:accountant&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;accountant&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:fire_engineer&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;fire_engineer&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:environmental_impact_assessment_engineer&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;environmental_impact_assessment_engineer&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:tax_accountant&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;tax_accountant&amp;quot;</span>),&quot;,&quot;    CustomCEVALEvaluationTask(name=<span class=\&quot;hljs-string\&quot;>&amp;quot;ceval:physician&amp;quot;</span>, hf_subset=<span class=\&quot;hljs-string\&quot;>&amp;quot;physician&amp;quot;</span>),&quot;,&quot;]&quot;,&quot;&quot;,&quot;ceval_subject_mapping = {&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;computer_network&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Computer Network&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;计算机网络&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;STEM&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;operating_system&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Operating System&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;操作系统&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;STEM&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;computer_architecture&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Computer Architecture&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;计算机组成&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;STEM&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;college_programming&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;College Programming&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;大学编程&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;STEM&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;college_physics&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;College Physics&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;大学物理&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;STEM&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;college_chemistry&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;College Chemistry&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;大学化学&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;STEM&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;advanced_mathematics&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Advanced Mathematics&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;高等数学&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;STEM&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;probability_and_statistics&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Probability and Statistics&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;概率统计&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;STEM&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;discrete_mathematics&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Discrete Mathematics&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;离散数学&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;STEM&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;electrical_engineer&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Electrical Engineer&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;注册电气工程师&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;STEM&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;metrology_engineer&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Metrology Engineer&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;注册计量师&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;STEM&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;high_school_mathematics&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;High School Mathematics&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;高中数学&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;STEM&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;high_school_physics&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;High School Physics&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;高中物理&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;STEM&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;high_school_chemistry&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;High School Chemistry&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;高中化学&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;STEM&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;high_school_biology&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;High School Biology&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;高中生物&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;STEM&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;middle_school_mathematics&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Middle School Mathematics&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;初中数学&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;STEM&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;middle_school_biology&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Middle School Biology&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;初中生物&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;STEM&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;middle_school_physics&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Middle School Physics&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;初中物理&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;STEM&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;middle_school_chemistry&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Middle School Chemistry&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;初中化学&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;STEM&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;veterinary_medicine&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Veterinary Medicine&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;兽医学&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;STEM&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;college_economics&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;College Economics&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;大学经济学&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Social Science&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;business_administration&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Business Administration&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;工商管理&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Social Science&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;marxism&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Marxism&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;马克思主义基本原理&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Social Science&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;mao_zedong_thought&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Mao Zedong Thought&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;毛泽东思想和中国特色社会主义理论体系概论&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Social Science&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;education_science&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Education Science&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;教育学&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Social Science&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;teacher_qualification&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Teacher Qualification&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;教师资格&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Social Science&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;high_school_politics&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;High School Politics&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;高中政治&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Social Science&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;high_school_geography&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;High School Geography&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;高中地理&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Social Science&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;middle_school_politics&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Middle School Politics&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;初中政治&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Social Science&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;middle_school_geography&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Middle School Geography&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;初中地理&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Social Science&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;modern_chinese_history&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Modern Chinese History&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;近代史纲要&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Humanities&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;ideological_and_moral_cultivation&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Ideological and Moral Cultivation&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;思想道德修养与法律基础&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Humanities&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;logic&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Logic&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;逻辑学&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Humanities&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;law&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Law&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;法学&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Humanities&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;chinese_language_and_literature&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Chinese Language and Literature&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;中国语言文学&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Humanities&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;art_studies&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Art Studies&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;艺术学&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Humanities&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;professional_tour_guide&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Professional Tour Guide&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;导游资格&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Humanities&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;legal_professional&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Legal Professional&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;法律职业资格&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Humanities&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;high_school_chinese&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;High School Chinese&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;高中语文&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Humanities&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;high_school_history&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;High School History&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;高中历史&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Humanities&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;middle_school_history&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Middle School History&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;初中历史&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Humanities&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;civil_servant&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Civil Servant&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;公务员&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Other&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;sports_science&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Sports Science&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;体育学&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Other&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;plant_protection&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Plant Protection&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;植物保护&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Other&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;basic_medicine&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Basic Medicine&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;基础医学&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Other&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;clinical_medicine&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Clinical Medicine&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;临床医学&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Other&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;urban_and_rural_planner&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Urban and Rural Planner&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;注册城乡规划师&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Other&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;accountant&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Accountant&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;注册会计师&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Other&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;fire_engineer&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Fire Engineer&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;注册消防工程师&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Other&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;environmental_impact_assessment_engineer&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Environmental Impact Assessment Engineer&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;环境影响评价工程师&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Other&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;tax_accountant&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Tax Accountant&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;税务师&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Other&amp;#x27;</span>],&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;#x27;physician&amp;#x27;</span>: [<span class=\&quot;hljs-string\&quot;>&amp;#x27;Physician&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;医师资格&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27;Other&amp;#x27;</span>],&quot;,&quot;}&quot;,&quot;&quot;,&quot;<span class=\&quot;hljs-keyword\&quot;>def</span> <span class=\&quot;hljs-title function_\&quot;>ceval_prompt</span>(<span class=\&quot;hljs-params\&quot;>line, task_name: <span class=\&quot;hljs-built_in\&quot;>str</span> = <span class=\&quot;hljs-literal\&quot;>None</span></span>):&quot;,&quot;    <span class=\&quot;hljs-comment\&quot;># f&amp;quot;以下是中国关于{_ch_name}考试的单项选择题，请选出其中的正确答案。\\n{{question}}\\nA. {{A}}\\nB. {{B}}\\nC. {{C}}\\nD. {{D}}\\n答案: &amp;quot;</span>&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;quot;&amp;quot;&amp;quot;CEVAL prompt without letters&amp;quot;&amp;quot;&amp;quot;</span>&quot;,&quot;    topic = ceval_subject_mapping[line[<span class=\&quot;hljs-string\&quot;>&amp;#x27;subject&amp;#x27;</span>]][<span class=\&quot;hljs-number\&quot;>1</span>]&quot;,&quot;    prompt = <span class=\&quot;hljs-string\&quot;>f&amp;quot;以下是中国关于<span class=\&quot;hljs-subst\&quot;>{topic.replace(<span class=\&quot;hljs-string\&quot;>&amp;#x27;_&amp;#x27;</span>, <span class=\&quot;hljs-string\&quot;>&amp;#x27; &amp;#x27;</span>)}</span>考试的单项选择题，请选出其中的正确答案。\\n题目：&amp;quot;</span>&quot;,&quot;    prompt += line[<span class=\&quot;hljs-string\&quot;>&amp;quot;question&amp;quot;</span>] + <span class=\&quot;hljs-string\&quot;>&amp;quot;\\n答案：&amp;quot;</span>&quot;,&quot;    <span class=\&quot;hljs-comment\&quot;>#print(f&amp;quot;ceval_prompt={prompt}&amp;quot;)</span>&quot;,&quot;&quot;,&quot;    <span class=\&quot;hljs-keyword\&quot;>return</span> Doc(&quot;,&quot;        task_name=task_name,&quot;,&quot;        query=prompt,&quot;,&quot;        choices=[<span class=\&quot;hljs-string\&quot;>f&amp;quot; <span class=\&quot;hljs-subst\&quot;>{c}</span>&amp;quot;</span> <span class=\&quot;hljs-keyword\&quot;>for</span> c <span class=\&quot;hljs-keyword\&quot;>in</span> line[<span class=\&quot;hljs-string\&quot;>&amp;quot;choices&amp;quot;</span>]],&quot;,&quot;        gold_index=line[<span class=\&quot;hljs-string\&quot;>&amp;quot;answer&amp;quot;</span>],&quot;,&quot;        instruction=<span class=\&quot;hljs-literal\&quot;>None</span>,&quot;,&quot;    )&quot;,&quot;&quot;,&quot;CEVAL_STRING = [(t, <span class=\&quot;hljs-string\&quot;>f&amp;quot;custom|<span class=\&quot;hljs-subst\&quot;>{t.name}</span>|0|1&amp;quot;</span>) <span class=\&quot;hljs-keyword\&quot;>for</span> t <span class=\&quot;hljs-keyword\&quot;>in</span> CEVAL_TASKS]&quot;,&quot;_TASKS_STRINGS.extend(CEVAL_STRING)&quot;,&quot;_TASKS += CEVAL_TASKS&quot;,&quot;<span class=\&quot;hljs-built_in\&quot;>print</span>(<span class=\&quot;hljs-string\&quot;>f&amp;#x27;<span class=\&quot;hljs-subst\&quot;>{<span class=\&quot;hljs-string\&quot;>&amp;quot;,&amp;quot;</span>.join([t[<span class=\&quot;hljs-number\&quot;>1</span>] <span class=\&quot;hljs-keyword\&quot;>for</span> t <span class=\&quot;hljs-keyword\&quot;>in</span> CEVAL_STRING])}</span>&amp;#x27;</span>)&quot;,&quot;&quot;,&quot;<span class=\&quot;hljs-comment\&quot;>############################################################################################################################################################</span>&quot;,&quot;&quot;,&quot;<span class=\&quot;hljs-comment\&quot;># common sense reasoning + mmlu</span>&quot;,&quot;EARLY_SIGNAL_TASKS = <span class=\&quot;hljs-string\&quot;>&amp;quot;,&amp;quot;</span>.join([t[<span class=\&quot;hljs-number\&quot;>1</span>] <span class=\&quot;hljs-keyword\&quot;>for</span> t <span class=\&quot;hljs-keyword\&quot;>in</span> COMMON_SENSE_REASONING_STRING] + [t[<span class=\&quot;hljs-number\&quot;>1</span>] <span class=\&quot;hljs-keyword\&quot;>for</span> t <span class=\&quot;hljs-keyword\&quot;>in</span> MMLU_STRING] + [t[<span class=\&quot;hljs-number\&quot;>1</span>] <span class=\&quot;hljs-keyword\&quot;>for</span> t <span class=\&quot;hljs-keyword\&quot;>in</span> CMMLU_STRING])&quot;,&quot;&quot;,&quot;<span class=\&quot;hljs-comment\&quot;># Convert to dict for lighteval</span>&quot;,&quot;TASKS_TABLE = [task.as_dict() <span class=\&quot;hljs-keyword\&quot;>for</span> task <span class=\&quot;hljs-keyword\&quot;>in</span> _TASKS]&quot;,&quot;<span class=\&quot;hljs-comment\&quot;># You can have a few pre-organised groups of tasks</span>&quot;,&quot;TASKS_GROUPS = {&quot;,&quot;    <span class=\&quot;hljs-string\&quot;>&amp;quot;early-signal&amp;quot;</span>: EARLY_SIGNAL_TASKS,&quot;,&quot;}&quot;,&quot;&quot;],&quot;lineSelectorClass&quot;:&quot;blob-line&quot;,&quot;context&quot;:{&quot;repo&quot;:{&quot;name&quot;:&quot;BAAI/CCI3-HQ&quot;,&quot;type&quot;:&quot;dataset&quot;},&quot;revision&quot;:&quot;b2a6a0eae226a8d877d62c414a890028d3c6f507&quot;,&quot;path&quot;:&quot;lighteval_tasks_v2.py&quot;}}">

<div class="relative text-sm"><div class="overflow-x-auto"><table class="min-w-full border-collapse font-mono"><tbody><tr class="" id="L1">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="1"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-comment"># ruff: noqa: F405, F403, F401</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L2">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="2"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-string">&quot;&quot;&quot;</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L3">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="3"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-string">Custom evaluation tasks for lighteval</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L4">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="4"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-string"></span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L5">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="5"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-string">Do note that we ran the evals with `max_samples=1000` to speed up large evals.</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L6">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="6"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-string">Most custom prompt changes were in an attempt to improve signal for small models in general.</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L7">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="7"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-string"></span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L8">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="8"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-string">This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L9">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="9"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-string"></span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L10">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="10"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-string">Example usage (lighteval_tasks.py is the path to this file):</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L11">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="11"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-string">===================</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L12">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="12"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-string">accelerate launch --num_processes=1 lighteval/run_evals_accelerate.py --model_args=&quot;pretrained=HuggingFaceFW/ablation-model-fineweb-edu&quot; \</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L13">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="13"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-string">    --custom_tasks &quot;lighteval_tasks.py&quot; --output_dir [OUTPUTPATH] --max_samples 1000 \ </span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L14">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="14"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-string">    --tasks &quot;custom|hellaswag|0|1,custom|winogrande|0|1,custom|piqa|0|1,custom|siqa|0|1,custom|openbookqa|0|1,custom|arc:easy|0|1,custom|arc:challenge|0|1,custom|commonsense_qa|0|1,custom|mmlu:abstract_algebra|0|1,custom|mmlu:anatomy|0|1,custom|mmlu:astronomy|0|1,custom|mmlu:business_ethics|0|1,custom|mmlu:clinical_knowledge|0|1,custom|mmlu:college_biology|0|1,custom|mmlu:college_chemistry|0|1,custom|mmlu:college_computer_science|0|1,custom|mmlu:college_mathematics|0|1,custom|mmlu:college_medicine|0|1,custom|mmlu:college_physics|0|1,custom|mmlu:computer_security|0|1,custom|mmlu:conceptual_physics|0|1,custom|mmlu:econometrics|0|1,custom|mmlu:electrical_engineering|0|1,custom|mmlu:elementary_mathematics|0|1,custom|mmlu:formal_logic|0|1,custom|mmlu:global_facts|0|1,custom|mmlu:high_school_biology|0|1,custom|mmlu:high_school_chemistry|0|1,custom|mmlu:high_school_computer_science|0|1,custom|mmlu:high_school_european_history|0|1,custom|mmlu:high_school_geography|0|1,custom|mmlu:high_school_government_and_politics|0|1,custom|mmlu:high_school_macroeconomics|0|1,custom|mmlu:high_school_mathematics|0|1,custom|mmlu:high_school_microeconomics|0|1,custom|mmlu:high_school_physics|0|1,custom|mmlu:high_school_psychology|0|1,custom|mmlu:high_school_statistics|0|1,custom|mmlu:high_school_us_history|0|1,custom|mmlu:high_school_world_history|0|1,custom|mmlu:human_aging|0|1,custom|mmlu:human_sexuality|0|1,custom|mmlu:international_law|0|1,custom|mmlu:jurisprudence|0|1,custom|mmlu:logical_fallacies|0|1,custom|mmlu:machine_learning|0|1,custom|mmlu:management|0|1,custom|mmlu:marketing|0|1,custom|mmlu:medical_genetics|0|1,custom|mmlu:miscellaneous|0|1,custom|mmlu:moral_disputes|0|1,custom|mmlu:moral_scenarios|0|1,custom|mmlu:nutrition|0|1,custom|mmlu:philosophy|0|1,custom|mmlu:prehistory|0|1,custom|mmlu:professional_accounting|0|1,custom|mmlu:professional_law|0|1,custom|mmlu:professional_medicine|0|1,custom|mmlu:professional_psychology|0|1,custom|mmlu:public_relations|0|1,custom|mmlu:security_studies|0|1,custom|mmlu:sociology|0|1,custom|mmlu:us_foreign_policy|0|1,custom|mmlu:virology|0|1,custom|mmlu:world_religions|0|1&quot;</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L15">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="15"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-string">===================</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L16">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="16"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-string">custom|cmmlu:agronomy|0|1,custom|cmmlu:anatomy|0|1,custom|cmmlu:ancient_chinese|0|1,custom|cmmlu:arts|0|1,custom|cmmlu:astronomy|0|1,custom|cmmlu:business_ethics|0|1,custom|cmmlu:chinese_civil_service_exam|0|1,custom|cmmlu:chinese_driving_rule|0|1,custom|cmmlu:chinese_food_culture|0|1,custom|cmmlu:chinese_foreign_policy|0|1,custom|cmmlu:chinese_history|0|1,custom|cmmlu:chinese_literature|0|1,custom|cmmlu:chinese_teacher_qualification|0|1,custom|cmmlu:clinical_knowledge|0|1,custom|cmmlu:college_actuarial_science|0|1,custom|cmmlu:college_education|0|1,custom|cmmlu:college_engineering_hydrology|0|1,custom|cmmlu:college_law|0|1,custom|cmmlu:college_mathematics|0|1,custom|cmmlu:college_medical_statistics|0|1,custom|cmmlu:college_medicine|0|1,custom|cmmlu:computer_science|0|1,custom|cmmlu:computer_security|0|1,custom|cmmlu:conceptual_physics|0|1,custom|cmmlu:construction_project_management|0|1,custom|cmmlu:economics|0|1,custom|cmmlu:education|0|1,custom|cmmlu:electrical_engineering|0|1,custom|cmmlu:elementary_chinese|0|1,custom|cmmlu:elementary_commonsense|0|1,custom|cmmlu:elementary_information_and_technology|0|1,custom|cmmlu:elementary_mathematics|0|1,custom|cmmlu:ethnology|0|1,custom|cmmlu:food_science|0|1,custom|cmmlu:genetics|0|1,custom|cmmlu:global_facts|0|1,custom|cmmlu:high_school_biology|0|1,custom|cmmlu:high_school_chemistry|0|1,custom|cmmlu:high_school_geography|0|1,custom|cmmlu:high_school_mathematics|0|1,custom|cmmlu:high_school_physics|0|1,custom|cmmlu:high_school_politics|0|1,custom|cmmlu:human_sexuality|0|1,custom|cmmlu:international_law|0|1,custom|cmmlu:journalism|0|1,custom|cmmlu:jurisprudence|0|1,custom|cmmlu:legal_and_moral_basis|0|1,custom|cmmlu:logical|0|1,custom|cmmlu:machine_learning|0|1,custom|cmmlu:management|0|1,custom|cmmlu:marketing|0|1,custom|cmmlu:marxist_theory|0|1,custom|cmmlu:modern_chinese|0|1,custom|cmmlu:nutrition|0|1,custom|cmmlu:philosophy|0|1,custom|cmmlu:professional_accounting|0|1,custom|cmmlu:professional_law|0|1,custom|cmmlu:professional_medicine|0|1,custom|cmmlu:professional_psychology|0|1,custom|cmmlu:public_relations|0|1,custom|cmmlu:security_study|0|1,custom|cmmlu:sociology|0|1,custom|cmmlu:sports_science|0|1,custom|cmmlu:traditional_chinese_medicine|0|1,custom|cmmlu:virology|0|1,custom|cmmlu:world_history|0|1,custom|cmmlu:world_religions|0|1</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L17">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="17"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-string">===================</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L18">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="18"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-string">custom|ceval:computer_network|0|1,custom|ceval:operating_system|0|1,custom|ceval:computer_architecture|0|1,custom|ceval:college_programming|0|1,custom|ceval:college_physics|0|1,custom|ceval:college_chemistry|0|1,custom|ceval:advanced_mathematics|0|1,custom|ceval:probability_and_statistics|0|1,custom|ceval:discrete_mathematics|0|1,custom|ceval:electrical_engineer|0|1,custom|ceval:metrology_engineer|0|1,custom|ceval:high_school_mathematics|0|1,custom|ceval:high_school_physics|0|1,custom|ceval:high_school_chemistry|0|1,custom|ceval:high_school_biology|0|1,custom|ceval:middle_school_mathematics|0|1,custom|ceval:middle_school_biology|0|1,custom|ceval:middle_school_physics|0|1,custom|ceval:middle_school_chemistry|0|1,custom|ceval:veterinary_medicine|0|1,custom|ceval:college_economics|0|1,custom|ceval:business_administration|0|1,custom|ceval:marxism|0|1,custom|ceval:mao_zedong_thought|0|1,custom|ceval:education_science|0|1,custom|ceval:teacher_qualification|0|1,custom|ceval:high_school_politics|0|1,custom|ceval:high_school_geography|0|1,custom|ceval:middle_school_politics|0|1,custom|ceval:middle_school_geography|0|1,custom|ceval:modern_chinese_history|0|1,custom|ceval:ideological_and_moral_cultivation|0|1,custom|ceval:logic|0|1,custom|ceval:law|0|1,custom|ceval:chinese_language_and_literature|0|1,custom|ceval:art_studies|0|1,custom|ceval:professional_tour_guide|0|1,custom|ceval:legal_professional|0|1,custom|ceval:high_school_chinese|0|1,custom|ceval:high_school_history|0|1,custom|ceval:middle_school_history|0|1,custom|ceval:civil_servant|0|1,custom|ceval:sports_science|0|1,custom|ceval:plant_protection|0|1,custom|ceval:basic_medicine|0|1,custom|ceval:clinical_medicine|0|1,custom|ceval:urban_and_rural_planner|0|1,custom|ceval:accountant|0|1,custom|ceval:fire_engineer|0|1,custom|ceval:environmental_impact_assessment_engineer|0|1,custom|ceval:tax_accountant|0|1,custom|ceval:physician|0|1</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L19">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="19"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-string">===================</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L20">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="20"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-string"></span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L21">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="21"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-string">More info here: https://github.com/huggingface/lighteval?tab=readme-ov-file#evaluate-a-model-on-extended-community-or-custom-tasks</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L22">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="22"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-string">For more info on differences between MMLU implementations: https://huggingface.co/blog/open-llm-leaderboard-mmlu#1001-flavors-of-mmlu</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L23">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="23"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-string">In particular, the default leaderboard MMLU implementation (which uses &quot;A&quot;, &quot;B&quot;, etc as answer targets) gives generally random results on small/non instruction tuned models.</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L24">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="24"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-string">Instead, we use the full MMLU answer as the target.</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L25">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="25"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-string">&quot;&quot;&quot;</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L26">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="26"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-keyword">import</span> re<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L27">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="27"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-keyword">from</span> typing <span class="hljs-keyword">import</span> <span class="hljs-type">List</span>, <span class="hljs-type">Tuple</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L28">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="28"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L29">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="29"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-keyword">from</span> lighteval.metrics <span class="hljs-keyword">import</span> Metrics<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L30">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="30"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-keyword">from</span> lighteval.tasks.lighteval_task <span class="hljs-keyword">import</span> LightevalTaskConfig<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L31">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="31"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-keyword">from</span> lighteval.tasks.requests <span class="hljs-keyword">import</span> Doc<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L32">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="32"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-keyword">from</span> lighteval.tasks.tasks_prompt_formatting <span class="hljs-keyword">import</span> LETTER_INDICES<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L33">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="33"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L34">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="34"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->_TASKS_STRINGS: <span class="hljs-type">List</span>[<span class="hljs-type">Tuple</span>[LightevalTaskConfig, <span class="hljs-built_in">str</span>]] = []<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L35">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="35"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->_TASKS: <span class="hljs-type">List</span>[LightevalTaskConfig] = []<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L36">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="36"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L37">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="37"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-comment">## COMMON_SENSE_REASONING_TASKS ##</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L38">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="38"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->COMMON_SENSE_REASONING_TASKS = [<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L39">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="39"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    LightevalTaskConfig(<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L40">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="40"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        name=<span class="hljs-string">&quot;hellaswag&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L41">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="41"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        prompt_function=<span class="hljs-string">&quot;hellaswag_prompt&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L42">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="42"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        hf_repo=<span class="hljs-string">&quot;hellaswag&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L43">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="43"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        hf_subset=<span class="hljs-string">&quot;default&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L44">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="44"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        metric=[<span class="hljs-string">&quot;loglikelihood_acc&quot;</span>, <span class="hljs-string">&quot;loglikelihood_acc_norm_nospace&quot;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L45">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="45"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    ),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L46">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="46"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    LightevalTaskConfig(<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L47">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="47"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        name=<span class="hljs-string">&quot;winogrande&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L48">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="48"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        prompt_function=<span class="hljs-string">&quot;winogrande&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L49">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="49"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        hf_repo=<span class="hljs-string">&quot;winogrande&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L50">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="50"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        hf_subset=<span class="hljs-string">&quot;winogrande_xl&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L51">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="51"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        metric=[<span class="hljs-string">&quot;loglikelihood_acc&quot;</span>, <span class="hljs-string">&quot;loglikelihood_acc_norm_nospace&quot;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L52">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="52"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    ),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L53">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="53"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    LightevalTaskConfig(<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L54">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="54"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        name=<span class="hljs-string">&quot;piqa&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L55">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="55"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        prompt_function=<span class="hljs-string">&quot;piqa_harness&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L56">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="56"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        hf_repo=<span class="hljs-string">&quot;piqa&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L57">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="57"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        hf_subset=<span class="hljs-string">&quot;plain_text&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L58">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="58"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        metric=[<span class="hljs-string">&quot;loglikelihood_acc&quot;</span>, <span class="hljs-string">&quot;loglikelihood_acc_norm_nospace&quot;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L59">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="59"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    ),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L60">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="60"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    LightevalTaskConfig(<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L61">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="61"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        name=<span class="hljs-string">&quot;siqa&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L62">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="62"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        prompt_function=<span class="hljs-string">&quot;siqa_prompt&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L63">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="63"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        hf_repo=<span class="hljs-string">&quot;lighteval/siqa&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L64">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="64"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        hf_subset=<span class="hljs-string">&quot;default&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L65">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="65"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        hf_avail_splits=[<span class="hljs-string">&quot;train&quot;</span>, <span class="hljs-string">&quot;validation&quot;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L66">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="66"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        metric=[<span class="hljs-string">&quot;loglikelihood_acc&quot;</span>, <span class="hljs-string">&quot;loglikelihood_acc_norm_nospace&quot;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L67">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="67"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    ),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L68">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="68"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    LightevalTaskConfig(<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L69">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="69"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        name=<span class="hljs-string">&quot;openbookqa&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L70">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="70"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        prompt_function=<span class="hljs-string">&quot;openbookqa&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L71">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="71"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        hf_repo=<span class="hljs-string">&quot;openbookqa&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L72">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="72"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        hf_subset=<span class="hljs-string">&quot;main&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L73">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="73"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        metric=[<span class="hljs-string">&quot;loglikelihood_acc&quot;</span>, <span class="hljs-string">&quot;loglikelihood_acc_norm_nospace&quot;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L74">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="74"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    ),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L75">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="75"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    LightevalTaskConfig(<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L76">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="76"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        name=<span class="hljs-string">&quot;arc:easy&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L77">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="77"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        prompt_function=<span class="hljs-string">&quot;arc&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L78">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="78"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        hf_repo=<span class="hljs-string">&quot;ai2_arc&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L79">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="79"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        hf_subset=<span class="hljs-string">&quot;ARC-Easy&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L80">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="80"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        evaluation_splits=[<span class="hljs-string">&quot;test&quot;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L81">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="81"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        generation_size=<span class="hljs-number">1</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L82">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="82"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        metric=[<span class="hljs-string">&quot;loglikelihood_acc&quot;</span>, <span class="hljs-string">&quot;loglikelihood_acc_norm_nospace&quot;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L83">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="83"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    ),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L84">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="84"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    LightevalTaskConfig(<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L85">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="85"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        name=<span class="hljs-string">&quot;arc:challenge&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L86">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="86"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        prompt_function=<span class="hljs-string">&quot;arc&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L87">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="87"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        hf_repo=<span class="hljs-string">&quot;ai2_arc&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L88">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="88"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        hf_subset=<span class="hljs-string">&quot;ARC-Challenge&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L89">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="89"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        evaluation_splits=[<span class="hljs-string">&quot;test&quot;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L90">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="90"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        generation_size=<span class="hljs-number">1</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L91">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="91"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        metric=[<span class="hljs-string">&quot;loglikelihood_acc&quot;</span>, <span class="hljs-string">&quot;loglikelihood_acc_norm_nospace&quot;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L92">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="92"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    ),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L93">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="93"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    LightevalTaskConfig(<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L94">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="94"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        name=<span class="hljs-string">&quot;commonsense_qa&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L95">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="95"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        prompt_function=<span class="hljs-string">&quot;commonsense_qa_prompt&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L96">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="96"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        hf_repo=<span class="hljs-string">&quot;commonsense_qa&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L97">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="97"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        hf_subset=<span class="hljs-string">&quot;default&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L98">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="98"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        metric=[<span class="hljs-string">&quot;loglikelihood_acc&quot;</span>, <span class="hljs-string">&quot;loglikelihood_acc_norm_nospace&quot;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L99">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="99"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    ),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L100">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="100"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->]<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L101">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="101"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L102">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="102"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L103">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="103"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-keyword">def</span> <span class="hljs-title function_">commonsense_qa_prompt</span>(<span class="hljs-params">line, task_name: <span class="hljs-built_in">str</span> = <span class="hljs-literal">None</span></span>):<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L104">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="104"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-keyword">return</span> Doc(<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L105">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="105"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        task_name=task_name,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L106">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="106"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        query=line[<span class="hljs-string">&quot;question&quot;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L107">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="107"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        choices=[<span class="hljs-string">f&quot; <span class="hljs-subst">{c}</span>&quot;</span> <span class="hljs-keyword">for</span> c <span class="hljs-keyword">in</span> line[<span class="hljs-string">&quot;choices&quot;</span>][<span class="hljs-string">&quot;text&quot;</span>]],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L108">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="108"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        gold_index=LETTER_INDICES.index(line[<span class="hljs-string">&quot;answerKey&quot;</span>].strip()),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L109">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="109"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        instruction=<span class="hljs-string">&quot;&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L110">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="110"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    )<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L111">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="111"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L112">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="112"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L113">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="113"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-keyword">def</span> <span class="hljs-title function_">siqa_prompt</span>(<span class="hljs-params">line, task_name: <span class="hljs-built_in">str</span> = <span class="hljs-literal">None</span></span>):<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L114">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="114"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-keyword">return</span> Doc(<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L115">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="115"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        task_name=task_name,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L116">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="116"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        query=line[<span class="hljs-string">&quot;context&quot;</span>] + <span class="hljs-string">&quot; &quot;</span> + line[<span class="hljs-string">&quot;question&quot;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L117">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="117"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        choices=[<span class="hljs-string">f&quot; <span class="hljs-subst">{c}</span>&quot;</span> <span class="hljs-keyword">for</span> c <span class="hljs-keyword">in</span> [line[<span class="hljs-string">&quot;answerA&quot;</span>], line[<span class="hljs-string">&quot;answerB&quot;</span>], line[<span class="hljs-string">&quot;answerC&quot;</span>]]],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L118">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="118"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        gold_index=<span class="hljs-built_in">int</span>(line[<span class="hljs-string">&quot;label&quot;</span>]) - <span class="hljs-number">1</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L119">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="119"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        instruction=<span class="hljs-string">&quot;&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L120">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="120"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    )<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L121">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="121"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L122">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="122"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L123">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="123"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-keyword">def</span> <span class="hljs-title function_">hellaswag_prompt</span>(<span class="hljs-params">line, task_name: <span class="hljs-built_in">str</span> = <span class="hljs-literal">None</span></span>):<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L124">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="124"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-keyword">def</span> <span class="hljs-title function_">preprocess</span>(<span class="hljs-params">text</span>):<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L125">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="125"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        <span class="hljs-string">&quot;&quot;&quot;Comes from AiHarness&quot;&quot;&quot;</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L126">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="126"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        <span class="hljs-comment"># text = text.strip()</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L127">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="127"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        <span class="hljs-comment"># <span class="hljs-doctag">NOTE:</span> Brackets are artifacts of the WikiHow dataset portion of HellaSwag.</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L128">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="128"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        text = text.replace(<span class="hljs-string">&quot; [title]&quot;</span>, <span class="hljs-string">&quot;. &quot;</span>)<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L129">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="129"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        text = re.sub(<span class="hljs-string">&quot;\\[.*?\\]&quot;</span>, <span class="hljs-string">&quot;&quot;</span>, text)<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L130">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="130"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        text = text.replace(<span class="hljs-string">&quot;  &quot;</span>, <span class="hljs-string">&quot; &quot;</span>)<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L131">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="131"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        <span class="hljs-keyword">return</span> text<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L132">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="132"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L133">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="133"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    ctx = <span class="hljs-string">f&quot;<span class="hljs-subst">{line[<span class="hljs-string">&#x27;ctx_a&#x27;</span>]}</span> <span class="hljs-subst">{line[<span class="hljs-string">&#x27;ctx_b&#x27;</span>].capitalize()}</span> &quot;</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L134">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="134"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-keyword">return</span> Doc(<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L135">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="135"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        task_name=task_name,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L136">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="136"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        query=preprocess(line[<span class="hljs-string">&quot;activity_label&quot;</span>] + <span class="hljs-string">&quot;: &quot;</span> + ctx),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L137">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="137"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        choices=[<span class="hljs-string">&quot; &quot;</span> + preprocess(ending) <span class="hljs-keyword">for</span> ending <span class="hljs-keyword">in</span> line[<span class="hljs-string">&quot;endings&quot;</span>]],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L138">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="138"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        gold_index=<span class="hljs-built_in">int</span>(line[<span class="hljs-string">&quot;label&quot;</span>]) <span class="hljs-keyword">if</span> line[<span class="hljs-string">&quot;label&quot;</span>] != <span class="hljs-string">&quot;&quot;</span> <span class="hljs-keyword">else</span> -<span class="hljs-number">1</span>,  <span class="hljs-comment"># -1 for test</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L139">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="139"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        <span class="hljs-comment"># &quot;metric&quot;: &quot;choices_loglikelihood&quot;,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L140">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="140"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    )<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L141">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="141"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L142">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="142"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L143">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="143"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-comment"># 0 short for common sense</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L144">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="144"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->COMMON_SENSE_REASONING_STRING = [(t, <span class="hljs-string">f&quot;custom|<span class="hljs-subst">{t.name}</span>|0|1&quot;</span>) <span class="hljs-keyword">for</span> t <span class="hljs-keyword">in</span> COMMON_SENSE_REASONING_TASKS]<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L145">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="145"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->_TASKS_STRINGS.extend(COMMON_SENSE_REASONING_STRING)<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L146">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="146"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->_TASKS += COMMON_SENSE_REASONING_TASKS<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L147">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="147"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L148">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="148"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-comment">## MMLU ##</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L149">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="149"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-keyword">class</span> <span class="hljs-title class_">CustomMMLUEvaluationTask</span>(<span class="hljs-title class_ inherited__">LightevalTaskConfig</span>):<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L150">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="150"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-keyword">def</span> <span class="hljs-title function_">__init__</span>(<span class="hljs-params"></span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L151">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="151"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        self,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L152">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="152"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        name,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L153">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="153"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        prompt_function=<span class="hljs-string">&quot;mmlu_prompt&quot;</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L154">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="154"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        hf_repo=<span class="hljs-string">&quot;lighteval/mmlu&quot;</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L155">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="155"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        hf_subset=<span class="hljs-literal">None</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L156">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="156"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        <span class="hljs-comment">#  metric=[Metrics.loglikelihood_acc_single_token],</span></span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L157">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="157"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L158">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="158"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        hf_avail_splits=<span class="hljs-literal">None</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L159">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="159"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        evaluation_splits=[<span class="hljs-string">&quot;test&quot;</span>],</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L160">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="160"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        few_shots_split=<span class="hljs-string">&quot;dev&quot;</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L161">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="161"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        few_shots_select=<span class="hljs-literal">None</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L162">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="162"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        suite=<span class="hljs-literal">None</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L163">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="163"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        generation_size=-<span class="hljs-number">1</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L164">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="164"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        stop_sequence=<span class="hljs-literal">None</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L165">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="165"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        output_regex=<span class="hljs-literal">None</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L166">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="166"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        frozen=<span class="hljs-literal">False</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L167">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="167"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">    </span>):<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L168">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="168"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        <span class="hljs-built_in">super</span>().__init__(<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L169">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="169"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            name=name,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L170">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="170"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            prompt_function=prompt_function,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L171">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="171"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            hf_repo=hf_repo,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L172">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="172"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            hf_subset=hf_subset,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L173">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="173"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            metric=metric,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L174">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="174"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            hf_avail_splits=hf_avail_splits,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L175">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="175"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            evaluation_splits=evaluation_splits,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L176">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="176"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            few_shots_split=few_shots_split,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L177">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="177"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            few_shots_select=few_shots_select,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L178">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="178"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            suite=suite,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L179">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="179"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            generation_size=generation_size,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L180">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="180"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            stop_sequence=stop_sequence,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L181">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="181"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            output_regex=output_regex,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L182">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="182"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            frozen=frozen,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L183">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="183"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        )<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L184">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="184"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L185">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="185"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L186">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="186"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->MMLU_TASKS = [<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L187">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="187"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:abstract_algebra&quot;</span>, hf_subset=<span class="hljs-string">&quot;abstract_algebra&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L188">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="188"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:anatomy&quot;</span>, hf_subset=<span class="hljs-string">&quot;anatomy&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L189">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="189"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:astronomy&quot;</span>, hf_subset=<span class="hljs-string">&quot;astronomy&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L190">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="190"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:business_ethics&quot;</span>, hf_subset=<span class="hljs-string">&quot;business_ethics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L191">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="191"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:clinical_knowledge&quot;</span>, hf_subset=<span class="hljs-string">&quot;clinical_knowledge&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L192">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="192"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:college_biology&quot;</span>, hf_subset=<span class="hljs-string">&quot;college_biology&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L193">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="193"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:college_chemistry&quot;</span>, hf_subset=<span class="hljs-string">&quot;college_chemistry&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L194">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="194"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:college_computer_science&quot;</span>, hf_subset=<span class="hljs-string">&quot;college_computer_science&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L195">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="195"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:college_mathematics&quot;</span>, hf_subset=<span class="hljs-string">&quot;college_mathematics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L196">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="196"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:college_medicine&quot;</span>, hf_subset=<span class="hljs-string">&quot;college_medicine&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L197">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="197"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:college_physics&quot;</span>, hf_subset=<span class="hljs-string">&quot;college_physics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L198">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="198"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:computer_security&quot;</span>, hf_subset=<span class="hljs-string">&quot;computer_security&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L199">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="199"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:conceptual_physics&quot;</span>, hf_subset=<span class="hljs-string">&quot;conceptual_physics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L200">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="200"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:econometrics&quot;</span>, hf_subset=<span class="hljs-string">&quot;econometrics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L201">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="201"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:electrical_engineering&quot;</span>, hf_subset=<span class="hljs-string">&quot;electrical_engineering&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L202">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="202"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:elementary_mathematics&quot;</span>, hf_subset=<span class="hljs-string">&quot;elementary_mathematics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L203">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="203"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:formal_logic&quot;</span>, hf_subset=<span class="hljs-string">&quot;formal_logic&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L204">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="204"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:global_facts&quot;</span>, hf_subset=<span class="hljs-string">&quot;global_facts&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L205">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="205"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:high_school_biology&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_biology&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L206">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="206"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:high_school_chemistry&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_chemistry&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L207">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="207"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:high_school_computer_science&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_computer_science&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L208">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="208"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:high_school_european_history&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_european_history&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L209">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="209"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:high_school_geography&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_geography&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L210">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="210"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L211">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="211"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        name=<span class="hljs-string">&quot;mmlu:high_school_government_and_politics&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_government_and_politics&quot;</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L212">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="212"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    ),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L213">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="213"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:high_school_macroeconomics&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_macroeconomics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L214">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="214"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:high_school_mathematics&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_mathematics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L215">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="215"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:high_school_microeconomics&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_microeconomics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L216">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="216"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:high_school_physics&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_physics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L217">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="217"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:high_school_psychology&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_psychology&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L218">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="218"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:high_school_statistics&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_statistics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L219">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="219"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:high_school_us_history&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_us_history&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L220">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="220"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:high_school_world_history&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_world_history&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L221">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="221"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:human_aging&quot;</span>, hf_subset=<span class="hljs-string">&quot;human_aging&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L222">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="222"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:human_sexuality&quot;</span>, hf_subset=<span class="hljs-string">&quot;human_sexuality&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L223">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="223"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:international_law&quot;</span>, hf_subset=<span class="hljs-string">&quot;international_law&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L224">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="224"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:jurisprudence&quot;</span>, hf_subset=<span class="hljs-string">&quot;jurisprudence&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L225">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="225"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:logical_fallacies&quot;</span>, hf_subset=<span class="hljs-string">&quot;logical_fallacies&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L226">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="226"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:machine_learning&quot;</span>, hf_subset=<span class="hljs-string">&quot;machine_learning&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L227">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="227"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:management&quot;</span>, hf_subset=<span class="hljs-string">&quot;management&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L228">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="228"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:marketing&quot;</span>, hf_subset=<span class="hljs-string">&quot;marketing&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L229">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="229"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:medical_genetics&quot;</span>, hf_subset=<span class="hljs-string">&quot;medical_genetics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L230">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="230"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:miscellaneous&quot;</span>, hf_subset=<span class="hljs-string">&quot;miscellaneous&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L231">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="231"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:moral_disputes&quot;</span>, hf_subset=<span class="hljs-string">&quot;moral_disputes&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L232">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="232"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:moral_scenarios&quot;</span>, hf_subset=<span class="hljs-string">&quot;moral_scenarios&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L233">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="233"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:nutrition&quot;</span>, hf_subset=<span class="hljs-string">&quot;nutrition&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L234">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="234"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:philosophy&quot;</span>, hf_subset=<span class="hljs-string">&quot;philosophy&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L235">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="235"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:prehistory&quot;</span>, hf_subset=<span class="hljs-string">&quot;prehistory&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L236">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="236"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:professional_accounting&quot;</span>, hf_subset=<span class="hljs-string">&quot;professional_accounting&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L237">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="237"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:professional_law&quot;</span>, hf_subset=<span class="hljs-string">&quot;professional_law&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L238">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="238"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:professional_medicine&quot;</span>, hf_subset=<span class="hljs-string">&quot;professional_medicine&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L239">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="239"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:professional_psychology&quot;</span>, hf_subset=<span class="hljs-string">&quot;professional_psychology&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L240">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="240"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:public_relations&quot;</span>, hf_subset=<span class="hljs-string">&quot;public_relations&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L241">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="241"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:security_studies&quot;</span>, hf_subset=<span class="hljs-string">&quot;security_studies&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L242">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="242"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:sociology&quot;</span>, hf_subset=<span class="hljs-string">&quot;sociology&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L243">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="243"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:us_foreign_policy&quot;</span>, hf_subset=<span class="hljs-string">&quot;us_foreign_policy&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L244">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="244"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:virology&quot;</span>, hf_subset=<span class="hljs-string">&quot;virology&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L245">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="245"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomMMLUEvaluationTask(name=<span class="hljs-string">&quot;mmlu:world_religions&quot;</span>, hf_subset=<span class="hljs-string">&quot;world_religions&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L246">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="246"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->]<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L247">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="247"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L248">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="248"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L249">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="249"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-keyword">def</span> <span class="hljs-title function_">mmlu_prompt</span>(<span class="hljs-params">line, task_name: <span class="hljs-built_in">str</span> = <span class="hljs-literal">None</span></span>):<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L250">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="250"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&quot;&quot;&quot;MMLU prompt without letters&quot;&quot;&quot;</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L251">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="251"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    topic = line[<span class="hljs-string">&quot;subject&quot;</span>]<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L252">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="252"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    prompt = <span class="hljs-string">f&quot;The following are questions about <span class="hljs-subst">{topic.replace(<span class="hljs-string">&#x27;_&#x27;</span>, <span class="hljs-string">&#x27; &#x27;</span>)}</span>.\nQuestion: &quot;</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L253">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="253"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    prompt += line[<span class="hljs-string">&quot;question&quot;</span>] + <span class="hljs-string">&quot;\nAnswer:&quot;</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L254">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="254"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-comment">#print(f&quot;mmlu_prompt={prompt}&quot;)</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L255">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="255"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L256">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="256"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-keyword">return</span> Doc(<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L257">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="257"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        task_name=task_name,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L258">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="258"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        query=prompt,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L259">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="259"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        choices=[<span class="hljs-string">f&quot; <span class="hljs-subst">{c}</span>&quot;</span> <span class="hljs-keyword">for</span> c <span class="hljs-keyword">in</span> line[<span class="hljs-string">&quot;choices&quot;</span>]],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L260">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="260"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        gold_index=line[<span class="hljs-string">&quot;answer&quot;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L261">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="261"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        instruction=<span class="hljs-string">f&quot;The following are questions about <span class="hljs-subst">{topic.replace(<span class="hljs-string">&#x27;_&#x27;</span>, <span class="hljs-string">&#x27; &#x27;</span>)}</span>.\n&quot;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L262">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="262"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    )<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L263">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="263"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L264">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="264"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L265">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="265"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->MMLU_STRING = [(t, <span class="hljs-string">f&quot;custom|<span class="hljs-subst">{t.name}</span>|0|1&quot;</span>) <span class="hljs-keyword">for</span> t <span class="hljs-keyword">in</span> MMLU_TASKS]<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L266">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="266"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->_TASKS_STRINGS.extend(MMLU_STRING)<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L267">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="267"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->_TASKS += MMLU_TASKS<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L268">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="268"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L269">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="269"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L270">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="270"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-comment">############################################################################################################################################################</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L271">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="271"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-comment">## CMMLU ##</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L272">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="272"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-keyword">class</span> <span class="hljs-title class_">CustomCMMLUEvaluationTask</span>(<span class="hljs-title class_ inherited__">LightevalTaskConfig</span>):<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L273">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="273"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-keyword">def</span> <span class="hljs-title function_">__init__</span>(<span class="hljs-params"></span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L274">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="274"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        self,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L275">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="275"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        name,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L276">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="276"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        prompt_function=<span class="hljs-string">&quot;cmmlu_prompt&quot;</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L277">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="277"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        hf_repo=<span class="hljs-string">&quot;ldwang/lighteval-cmmlu&quot;</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L278">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="278"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        hf_subset=<span class="hljs-literal">None</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L279">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="279"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        <span class="hljs-comment">#  metric=[Metrics.loglikelihood_acc_single_token],</span></span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L280">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="280"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L281">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="281"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        hf_avail_splits=<span class="hljs-literal">None</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L282">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="282"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        evaluation_splits=[<span class="hljs-string">&quot;test&quot;</span>],</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L283">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="283"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        few_shots_split=<span class="hljs-string">&quot;dev&quot;</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L284">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="284"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        few_shots_select=<span class="hljs-literal">None</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L285">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="285"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        suite=<span class="hljs-literal">None</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L286">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="286"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        generation_size=-<span class="hljs-number">1</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L287">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="287"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        stop_sequence=<span class="hljs-literal">None</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L288">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="288"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        output_regex=<span class="hljs-literal">None</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L289">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="289"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        frozen=<span class="hljs-literal">False</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L290">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="290"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">    </span>):<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L291">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="291"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        <span class="hljs-built_in">super</span>().__init__(<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L292">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="292"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            name=name,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L293">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="293"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            prompt_function=prompt_function,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L294">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="294"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            hf_repo=hf_repo,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L295">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="295"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            hf_subset=hf_subset,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L296">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="296"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            metric=metric,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L297">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="297"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            hf_avail_splits=hf_avail_splits,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L298">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="298"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            evaluation_splits=evaluation_splits,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L299">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="299"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            few_shots_split=few_shots_split,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L300">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="300"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            few_shots_select=few_shots_select,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L301">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="301"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            suite=suite,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L302">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="302"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            generation_size=generation_size,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L303">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="303"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            stop_sequence=stop_sequence,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L304">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="304"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            output_regex=output_regex,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L305">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="305"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            frozen=frozen,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L306">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="306"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            trust_dataset=<span class="hljs-literal">True</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L307">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="307"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        )<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L308">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="308"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L309">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="309"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L310">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="310"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->CMMLU_TASKS = [<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L311">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="311"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:agronomy&quot;</span>, hf_subset=<span class="hljs-string">&quot;agronomy&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L312">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="312"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:anatomy&quot;</span>, hf_subset=<span class="hljs-string">&quot;anatomy&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L313">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="313"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:ancient_chinese&quot;</span>, hf_subset=<span class="hljs-string">&quot;ancient_chinese&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L314">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="314"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:arts&quot;</span>, hf_subset=<span class="hljs-string">&quot;arts&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L315">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="315"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:astronomy&quot;</span>, hf_subset=<span class="hljs-string">&quot;astronomy&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L316">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="316"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:business_ethics&quot;</span>, hf_subset=<span class="hljs-string">&quot;business_ethics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L317">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="317"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:chinese_civil_service_exam&quot;</span>, hf_subset=<span class="hljs-string">&quot;chinese_civil_service_exam&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L318">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="318"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:chinese_driving_rule&quot;</span>, hf_subset=<span class="hljs-string">&quot;chinese_driving_rule&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L319">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="319"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:chinese_food_culture&quot;</span>, hf_subset=<span class="hljs-string">&quot;chinese_food_culture&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L320">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="320"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:chinese_foreign_policy&quot;</span>, hf_subset=<span class="hljs-string">&quot;chinese_foreign_policy&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L321">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="321"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:chinese_history&quot;</span>, hf_subset=<span class="hljs-string">&quot;chinese_history&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L322">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="322"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:chinese_literature&quot;</span>, hf_subset=<span class="hljs-string">&quot;chinese_literature&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L323">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="323"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:chinese_teacher_qualification&quot;</span>, hf_subset=<span class="hljs-string">&quot;chinese_teacher_qualification&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L324">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="324"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:clinical_knowledge&quot;</span>, hf_subset=<span class="hljs-string">&quot;clinical_knowledge&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L325">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="325"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:college_actuarial_science&quot;</span>, hf_subset=<span class="hljs-string">&quot;college_actuarial_science&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L326">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="326"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:college_education&quot;</span>, hf_subset=<span class="hljs-string">&quot;college_education&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L327">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="327"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:college_engineering_hydrology&quot;</span>, hf_subset=<span class="hljs-string">&quot;college_engineering_hydrology&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L328">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="328"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:college_law&quot;</span>, hf_subset=<span class="hljs-string">&quot;college_law&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L329">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="329"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:college_mathematics&quot;</span>, hf_subset=<span class="hljs-string">&quot;college_mathematics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L330">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="330"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:college_medical_statistics&quot;</span>, hf_subset=<span class="hljs-string">&quot;college_medical_statistics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L331">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="331"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:college_medicine&quot;</span>, hf_subset=<span class="hljs-string">&quot;college_medicine&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L332">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="332"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:computer_science&quot;</span>, hf_subset=<span class="hljs-string">&quot;computer_science&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L333">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="333"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:computer_security&quot;</span>, hf_subset=<span class="hljs-string">&quot;computer_security&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L334">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="334"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:conceptual_physics&quot;</span>, hf_subset=<span class="hljs-string">&quot;conceptual_physics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L335">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="335"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:construction_project_management&quot;</span>, hf_subset=<span class="hljs-string">&quot;construction_project_management&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L336">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="336"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:economics&quot;</span>, hf_subset=<span class="hljs-string">&quot;economics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L337">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="337"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:education&quot;</span>, hf_subset=<span class="hljs-string">&quot;education&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L338">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="338"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:electrical_engineering&quot;</span>, hf_subset=<span class="hljs-string">&quot;electrical_engineering&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L339">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="339"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:elementary_chinese&quot;</span>, hf_subset=<span class="hljs-string">&quot;elementary_chinese&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L340">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="340"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:elementary_commonsense&quot;</span>, hf_subset=<span class="hljs-string">&quot;elementary_commonsense&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L341">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="341"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:elementary_information_and_technology&quot;</span>, hf_subset=<span class="hljs-string">&quot;elementary_information_and_technology&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L342">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="342"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:elementary_mathematics&quot;</span>, hf_subset=<span class="hljs-string">&quot;elementary_mathematics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L343">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="343"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:ethnology&quot;</span>, hf_subset=<span class="hljs-string">&quot;ethnology&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L344">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="344"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:food_science&quot;</span>, hf_subset=<span class="hljs-string">&quot;food_science&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L345">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="345"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:genetics&quot;</span>, hf_subset=<span class="hljs-string">&quot;genetics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L346">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="346"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:global_facts&quot;</span>, hf_subset=<span class="hljs-string">&quot;global_facts&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L347">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="347"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:high_school_biology&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_biology&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L348">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="348"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:high_school_chemistry&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_chemistry&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L349">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="349"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:high_school_geography&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_geography&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L350">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="350"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:high_school_mathematics&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_mathematics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L351">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="351"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:high_school_physics&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_physics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L352">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="352"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:high_school_politics&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_politics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L353">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="353"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:human_sexuality&quot;</span>, hf_subset=<span class="hljs-string">&quot;human_sexuality&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L354">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="354"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:international_law&quot;</span>, hf_subset=<span class="hljs-string">&quot;international_law&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L355">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="355"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:journalism&quot;</span>, hf_subset=<span class="hljs-string">&quot;journalism&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L356">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="356"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:jurisprudence&quot;</span>, hf_subset=<span class="hljs-string">&quot;jurisprudence&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L357">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="357"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:legal_and_moral_basis&quot;</span>, hf_subset=<span class="hljs-string">&quot;legal_and_moral_basis&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L358">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="358"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:logical&quot;</span>, hf_subset=<span class="hljs-string">&quot;logical&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L359">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="359"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:machine_learning&quot;</span>, hf_subset=<span class="hljs-string">&quot;machine_learning&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L360">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="360"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:management&quot;</span>, hf_subset=<span class="hljs-string">&quot;management&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L361">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="361"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:marketing&quot;</span>, hf_subset=<span class="hljs-string">&quot;marketing&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L362">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="362"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:marxist_theory&quot;</span>, hf_subset=<span class="hljs-string">&quot;marxist_theory&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L363">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="363"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:modern_chinese&quot;</span>, hf_subset=<span class="hljs-string">&quot;modern_chinese&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L364">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="364"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:nutrition&quot;</span>, hf_subset=<span class="hljs-string">&quot;nutrition&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L365">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="365"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:philosophy&quot;</span>, hf_subset=<span class="hljs-string">&quot;philosophy&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L366">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="366"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:professional_accounting&quot;</span>, hf_subset=<span class="hljs-string">&quot;professional_accounting&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L367">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="367"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:professional_law&quot;</span>, hf_subset=<span class="hljs-string">&quot;professional_law&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L368">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="368"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:professional_medicine&quot;</span>, hf_subset=<span class="hljs-string">&quot;professional_medicine&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L369">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="369"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:professional_psychology&quot;</span>, hf_subset=<span class="hljs-string">&quot;professional_psychology&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L370">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="370"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:public_relations&quot;</span>, hf_subset=<span class="hljs-string">&quot;public_relations&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L371">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="371"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:security_study&quot;</span>, hf_subset=<span class="hljs-string">&quot;security_study&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L372">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="372"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:sociology&quot;</span>, hf_subset=<span class="hljs-string">&quot;sociology&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L373">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="373"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:sports_science&quot;</span>, hf_subset=<span class="hljs-string">&quot;sports_science&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L374">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="374"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:traditional_chinese_medicine&quot;</span>, hf_subset=<span class="hljs-string">&quot;traditional_chinese_medicine&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L375">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="375"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:virology&quot;</span>, hf_subset=<span class="hljs-string">&quot;virology&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L376">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="376"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:world_history&quot;</span>, hf_subset=<span class="hljs-string">&quot;world_history&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L377">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="377"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->		CustomCMMLUEvaluationTask(name=<span class="hljs-string">&quot;cmmlu:world_religions&quot;</span>, hf_subset=<span class="hljs-string">&quot;world_religions&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L378">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="378"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->]<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L379">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="379"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L380">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="380"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->cmmlu_subject_mapping = {<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L381">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="381"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;agronomy&#x27;</span>: <span class="hljs-string">&#x27;农学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L382">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="382"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;anatomy&#x27;</span>: <span class="hljs-string">&#x27;解剖学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L383">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="383"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;ancient_chinese&#x27;</span>: <span class="hljs-string">&#x27;古汉语&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L384">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="384"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;arts&#x27;</span>: <span class="hljs-string">&#x27;艺术学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L385">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="385"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;astronomy&#x27;</span>: <span class="hljs-string">&#x27;天文学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L386">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="386"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;business_ethics&#x27;</span>: <span class="hljs-string">&#x27;商业伦理&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L387">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="387"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;chinese_civil_service_exam&#x27;</span>: <span class="hljs-string">&#x27;中国公务员考试&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L388">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="388"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;chinese_driving_rule&#x27;</span>: <span class="hljs-string">&#x27;中国驾驶规则&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L389">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="389"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;chinese_food_culture&#x27;</span>: <span class="hljs-string">&#x27;中国饮食文化&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L390">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="390"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;chinese_foreign_policy&#x27;</span>: <span class="hljs-string">&#x27;中国外交政策&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L391">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="391"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;chinese_history&#x27;</span>: <span class="hljs-string">&#x27;中国历史&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L392">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="392"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;chinese_literature&#x27;</span>: <span class="hljs-string">&#x27;中国文学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L393">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="393"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;chinese_teacher_qualification&#x27;</span>: <span class="hljs-string">&#x27;中国教师资格&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L394">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="394"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;clinical_knowledge&#x27;</span>: <span class="hljs-string">&#x27;临床知识&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L395">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="395"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;college_actuarial_science&#x27;</span>: <span class="hljs-string">&#x27;大学精算学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L396">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="396"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;college_education&#x27;</span>: <span class="hljs-string">&#x27;大学教育学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L397">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="397"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;college_engineering_hydrology&#x27;</span>: <span class="hljs-string">&#x27;大学工程水文学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L398">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="398"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;college_law&#x27;</span>: <span class="hljs-string">&#x27;大学法律&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L399">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="399"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;college_mathematics&#x27;</span>: <span class="hljs-string">&#x27;大学数学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L400">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="400"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;college_medical_statistics&#x27;</span>: <span class="hljs-string">&#x27;大学医学统计&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L401">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="401"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;college_medicine&#x27;</span>: <span class="hljs-string">&#x27;大学医学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L402">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="402"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;computer_science&#x27;</span>: <span class="hljs-string">&#x27;计算机科学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L403">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="403"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;computer_security&#x27;</span>: <span class="hljs-string">&#x27;计算机安全&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L404">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="404"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;conceptual_physics&#x27;</span>: <span class="hljs-string">&#x27;概念物理学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L405">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="405"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;construction_project_management&#x27;</span>: <span class="hljs-string">&#x27;建设工程管理&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L406">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="406"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;economics&#x27;</span>: <span class="hljs-string">&#x27;经济学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L407">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="407"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;education&#x27;</span>: <span class="hljs-string">&#x27;教育学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L408">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="408"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;electrical_engineering&#x27;</span>: <span class="hljs-string">&#x27;电气工程&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L409">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="409"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;elementary_chinese&#x27;</span>: <span class="hljs-string">&#x27;小学语文&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L410">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="410"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;elementary_commonsense&#x27;</span>: <span class="hljs-string">&#x27;小学常识&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L411">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="411"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;elementary_information_and_technology&#x27;</span>: <span class="hljs-string">&#x27;小学信息技术&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L412">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="412"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;elementary_mathematics&#x27;</span>: <span class="hljs-string">&#x27;初等数学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L413">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="413"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;ethnology&#x27;</span>: <span class="hljs-string">&#x27;民族学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L414">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="414"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;food_science&#x27;</span>: <span class="hljs-string">&#x27;食品科学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L415">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="415"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;genetics&#x27;</span>: <span class="hljs-string">&#x27;遗传学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L416">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="416"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;global_facts&#x27;</span>: <span class="hljs-string">&#x27;全球事实&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L417">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="417"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;high_school_biology&#x27;</span>: <span class="hljs-string">&#x27;高中生物&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L418">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="418"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;high_school_chemistry&#x27;</span>: <span class="hljs-string">&#x27;高中化学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L419">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="419"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;high_school_geography&#x27;</span>: <span class="hljs-string">&#x27;高中地理&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L420">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="420"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;high_school_mathematics&#x27;</span>: <span class="hljs-string">&#x27;高中数学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L421">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="421"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;high_school_physics&#x27;</span>: <span class="hljs-string">&#x27;高中物理学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L422">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="422"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;high_school_politics&#x27;</span>: <span class="hljs-string">&#x27;高中政治&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L423">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="423"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;human_sexuality&#x27;</span>: <span class="hljs-string">&#x27;人类性行为&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L424">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="424"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;international_law&#x27;</span>: <span class="hljs-string">&#x27;国际法学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L425">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="425"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;journalism&#x27;</span>: <span class="hljs-string">&#x27;新闻学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L426">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="426"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;jurisprudence&#x27;</span>: <span class="hljs-string">&#x27;法理学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L427">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="427"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;legal_and_moral_basis&#x27;</span>: <span class="hljs-string">&#x27;法律与道德基础&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L428">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="428"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;logical&#x27;</span>: <span class="hljs-string">&#x27;逻辑学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L429">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="429"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;machine_learning&#x27;</span>: <span class="hljs-string">&#x27;机器学习&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L430">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="430"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;management&#x27;</span>: <span class="hljs-string">&#x27;管理学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L431">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="431"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;marketing&#x27;</span>: <span class="hljs-string">&#x27;市场营销&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L432">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="432"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;marxist_theory&#x27;</span>: <span class="hljs-string">&#x27;马克思主义理论&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L433">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="433"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;modern_chinese&#x27;</span>: <span class="hljs-string">&#x27;现代汉语&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L434">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="434"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;nutrition&#x27;</span>: <span class="hljs-string">&#x27;营养学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L435">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="435"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;philosophy&#x27;</span>: <span class="hljs-string">&#x27;哲学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L436">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="436"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;professional_accounting&#x27;</span>: <span class="hljs-string">&#x27;专业会计&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L437">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="437"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;professional_law&#x27;</span>: <span class="hljs-string">&#x27;专业法学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L438">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="438"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;professional_medicine&#x27;</span>: <span class="hljs-string">&#x27;专业医学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L439">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="439"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;professional_psychology&#x27;</span>: <span class="hljs-string">&#x27;专业心理学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L440">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="440"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;public_relations&#x27;</span>: <span class="hljs-string">&#x27;公共关系&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L441">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="441"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;security_study&#x27;</span>: <span class="hljs-string">&#x27;安全研究&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L442">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="442"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;sociology&#x27;</span>: <span class="hljs-string">&#x27;社会学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L443">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="443"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;sports_science&#x27;</span>: <span class="hljs-string">&#x27;体育学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L444">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="444"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;traditional_chinese_medicine&#x27;</span>: <span class="hljs-string">&#x27;中医中药&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L445">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="445"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;virology&#x27;</span>: <span class="hljs-string">&#x27;病毒学&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L446">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="446"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;world_history&#x27;</span>: <span class="hljs-string">&#x27;世界历史&#x27;</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L447">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="447"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;world_religions&#x27;</span>: <span class="hljs-string">&#x27;世界宗教&#x27;</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L448">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="448"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->}<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L449">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="449"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L450">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="450"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-keyword">def</span> <span class="hljs-title function_">cmmlu_prompt</span>(<span class="hljs-params">line, task_name: <span class="hljs-built_in">str</span> = <span class="hljs-literal">None</span></span>):<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L451">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="451"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-comment"># 以下是关于{_ch_name}的单项选择题，请直接给出正确答案的选项。\n题目：{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L452">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="452"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-comment"># 答案是: {{{answer}}}</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L453">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="453"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&quot;&quot;&quot;CMMLU prompt without letters&quot;&quot;&quot;</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L454">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="454"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    topic = cmmlu_subject_mapping[line[<span class="hljs-string">&#x27;subject&#x27;</span>]]<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L455">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="455"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    prompt = <span class="hljs-string">f&quot;以下是关于<span class="hljs-subst">{topic.replace(<span class="hljs-string">&#x27;_&#x27;</span>, <span class="hljs-string">&#x27; &#x27;</span>)}</span>的单项选择题，请直接给出正确答案的选项。\n题目：&quot;</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L456">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="456"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    prompt += line[<span class="hljs-string">&quot;question&quot;</span>] + <span class="hljs-string">&quot;\n答案是：&quot;</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L457">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="457"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-comment">#print(f&quot;cmmlu_prompt={prompt}&quot;)</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L458">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="458"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L459">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="459"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-keyword">return</span> Doc(<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L460">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="460"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        task_name=task_name,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L461">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="461"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        query=prompt,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L462">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="462"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        choices=[<span class="hljs-string">f&quot; <span class="hljs-subst">{c}</span>&quot;</span> <span class="hljs-keyword">for</span> c <span class="hljs-keyword">in</span> line[<span class="hljs-string">&quot;choices&quot;</span>]],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L463">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="463"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        gold_index=line[<span class="hljs-string">&quot;answer&quot;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L464">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="464"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        instruction=<span class="hljs-literal">None</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L465">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="465"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    )<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L466">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="466"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L467">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="467"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->CMMLU_STRING = [(t, <span class="hljs-string">f&quot;custom|<span class="hljs-subst">{t.name}</span>|0|1&quot;</span>) <span class="hljs-keyword">for</span> t <span class="hljs-keyword">in</span> CMMLU_TASKS]<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L468">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="468"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->_TASKS_STRINGS.extend(CMMLU_STRING)<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L469">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="469"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->_TASKS += CMMLU_TASKS<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L470">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="470"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-built_in">print</span>(<span class="hljs-string">f&#x27;<span class="hljs-subst">{<span class="hljs-string">&quot;,&quot;</span>.join([t[<span class="hljs-number">1</span>] <span class="hljs-keyword">for</span> t <span class="hljs-keyword">in</span> CMMLU_STRING])}</span>&#x27;</span>)<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L471">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="471"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L472">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="472"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-comment">############################################################################################################################################################</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L473">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="473"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-comment">## CEVAL ##</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L474">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="474"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-keyword">class</span> <span class="hljs-title class_">CustomCEVALEvaluationTask</span>(<span class="hljs-title class_ inherited__">LightevalTaskConfig</span>):<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L475">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="475"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-keyword">def</span> <span class="hljs-title function_">__init__</span>(<span class="hljs-params"></span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L476">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="476"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        self,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L477">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="477"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        name,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L478">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="478"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        prompt_function=<span class="hljs-string">&quot;ceval_prompt&quot;</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L479">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="479"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        hf_repo=<span class="hljs-string">&quot;ldwang/lighteval-ceval-exam&quot;</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L480">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="480"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        hf_subset=<span class="hljs-literal">None</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L481">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="481"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        <span class="hljs-comment">#  metric=[Metrics.loglikelihood_acc_single_token],</span></span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L482">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="482"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        metric=[Metrics.loglikelihood_acc, Metrics.loglikelihood_acc_norm_nospace],</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L483">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="483"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        hf_avail_splits=<span class="hljs-literal">None</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L484">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="484"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        evaluation_splits=[<span class="hljs-string">&quot;val&quot;</span>],</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L485">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="485"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        few_shots_split=<span class="hljs-string">&quot;dev&quot;</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L486">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="486"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        few_shots_select=<span class="hljs-literal">None</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L487">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="487"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        suite=<span class="hljs-literal">None</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L488">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="488"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        generation_size=-<span class="hljs-number">1</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L489">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="489"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        stop_sequence=<span class="hljs-literal">None</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L490">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="490"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        output_regex=<span class="hljs-literal">None</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L491">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="491"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">        frozen=<span class="hljs-literal">False</span>,</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L492">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="492"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-params">    </span>):<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L493">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="493"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        <span class="hljs-built_in">super</span>().__init__(<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L494">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="494"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            name=name,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L495">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="495"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            prompt_function=prompt_function,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L496">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="496"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            hf_repo=hf_repo,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L497">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="497"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            hf_subset=hf_subset,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L498">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="498"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            metric=metric,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L499">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="499"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            hf_avail_splits=hf_avail_splits,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L500">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="500"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            evaluation_splits=evaluation_splits,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L501">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="501"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            few_shots_split=few_shots_split,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L502">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="502"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            few_shots_select=few_shots_select,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L503">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="503"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            suite=suite,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L504">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="504"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            generation_size=generation_size,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L505">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="505"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            stop_sequence=stop_sequence,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L506">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="506"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            output_regex=output_regex,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L507">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="507"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            frozen=frozen,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L508">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="508"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->            trust_dataset=<span class="hljs-literal">True</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L509">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="509"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        )<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L510">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="510"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L511">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="511"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L512">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="512"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->CEVAL_TASKS = [<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L513">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="513"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:computer_network&quot;</span>, hf_subset=<span class="hljs-string">&quot;computer_network&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L514">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="514"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:operating_system&quot;</span>, hf_subset=<span class="hljs-string">&quot;operating_system&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L515">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="515"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:computer_architecture&quot;</span>, hf_subset=<span class="hljs-string">&quot;computer_architecture&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L516">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="516"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:college_programming&quot;</span>, hf_subset=<span class="hljs-string">&quot;college_programming&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L517">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="517"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:college_physics&quot;</span>, hf_subset=<span class="hljs-string">&quot;college_physics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L518">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="518"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:college_chemistry&quot;</span>, hf_subset=<span class="hljs-string">&quot;college_chemistry&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L519">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="519"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:advanced_mathematics&quot;</span>, hf_subset=<span class="hljs-string">&quot;advanced_mathematics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L520">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="520"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:probability_and_statistics&quot;</span>, hf_subset=<span class="hljs-string">&quot;probability_and_statistics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L521">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="521"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:discrete_mathematics&quot;</span>, hf_subset=<span class="hljs-string">&quot;discrete_mathematics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L522">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="522"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:electrical_engineer&quot;</span>, hf_subset=<span class="hljs-string">&quot;electrical_engineer&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L523">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="523"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:metrology_engineer&quot;</span>, hf_subset=<span class="hljs-string">&quot;metrology_engineer&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L524">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="524"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:high_school_mathematics&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_mathematics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L525">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="525"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:high_school_physics&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_physics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L526">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="526"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:high_school_chemistry&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_chemistry&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L527">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="527"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:high_school_biology&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_biology&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L528">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="528"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:middle_school_mathematics&quot;</span>, hf_subset=<span class="hljs-string">&quot;middle_school_mathematics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L529">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="529"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:middle_school_biology&quot;</span>, hf_subset=<span class="hljs-string">&quot;middle_school_biology&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L530">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="530"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:middle_school_physics&quot;</span>, hf_subset=<span class="hljs-string">&quot;middle_school_physics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L531">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="531"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:middle_school_chemistry&quot;</span>, hf_subset=<span class="hljs-string">&quot;middle_school_chemistry&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L532">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="532"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:veterinary_medicine&quot;</span>, hf_subset=<span class="hljs-string">&quot;veterinary_medicine&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L533">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="533"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:college_economics&quot;</span>, hf_subset=<span class="hljs-string">&quot;college_economics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L534">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="534"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:business_administration&quot;</span>, hf_subset=<span class="hljs-string">&quot;business_administration&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L535">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="535"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:marxism&quot;</span>, hf_subset=<span class="hljs-string">&quot;marxism&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L536">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="536"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:mao_zedong_thought&quot;</span>, hf_subset=<span class="hljs-string">&quot;mao_zedong_thought&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L537">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="537"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:education_science&quot;</span>, hf_subset=<span class="hljs-string">&quot;education_science&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L538">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="538"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:teacher_qualification&quot;</span>, hf_subset=<span class="hljs-string">&quot;teacher_qualification&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L539">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="539"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:high_school_politics&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_politics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L540">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="540"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:high_school_geography&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_geography&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L541">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="541"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:middle_school_politics&quot;</span>, hf_subset=<span class="hljs-string">&quot;middle_school_politics&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L542">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="542"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:middle_school_geography&quot;</span>, hf_subset=<span class="hljs-string">&quot;middle_school_geography&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L543">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="543"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:modern_chinese_history&quot;</span>, hf_subset=<span class="hljs-string">&quot;modern_chinese_history&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L544">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="544"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:ideological_and_moral_cultivation&quot;</span>, hf_subset=<span class="hljs-string">&quot;ideological_and_moral_cultivation&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L545">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="545"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:logic&quot;</span>, hf_subset=<span class="hljs-string">&quot;logic&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L546">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="546"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:law&quot;</span>, hf_subset=<span class="hljs-string">&quot;law&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L547">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="547"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:chinese_language_and_literature&quot;</span>, hf_subset=<span class="hljs-string">&quot;chinese_language_and_literature&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L548">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="548"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:art_studies&quot;</span>, hf_subset=<span class="hljs-string">&quot;art_studies&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L549">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="549"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:professional_tour_guide&quot;</span>, hf_subset=<span class="hljs-string">&quot;professional_tour_guide&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L550">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="550"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:legal_professional&quot;</span>, hf_subset=<span class="hljs-string">&quot;legal_professional&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L551">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="551"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:high_school_chinese&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_chinese&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L552">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="552"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:high_school_history&quot;</span>, hf_subset=<span class="hljs-string">&quot;high_school_history&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L553">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="553"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:middle_school_history&quot;</span>, hf_subset=<span class="hljs-string">&quot;middle_school_history&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L554">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="554"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:civil_servant&quot;</span>, hf_subset=<span class="hljs-string">&quot;civil_servant&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L555">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="555"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:sports_science&quot;</span>, hf_subset=<span class="hljs-string">&quot;sports_science&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L556">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="556"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:plant_protection&quot;</span>, hf_subset=<span class="hljs-string">&quot;plant_protection&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L557">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="557"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:basic_medicine&quot;</span>, hf_subset=<span class="hljs-string">&quot;basic_medicine&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L558">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="558"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:clinical_medicine&quot;</span>, hf_subset=<span class="hljs-string">&quot;clinical_medicine&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L559">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="559"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:urban_and_rural_planner&quot;</span>, hf_subset=<span class="hljs-string">&quot;urban_and_rural_planner&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L560">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="560"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:accountant&quot;</span>, hf_subset=<span class="hljs-string">&quot;accountant&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L561">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="561"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:fire_engineer&quot;</span>, hf_subset=<span class="hljs-string">&quot;fire_engineer&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L562">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="562"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:environmental_impact_assessment_engineer&quot;</span>, hf_subset=<span class="hljs-string">&quot;environmental_impact_assessment_engineer&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L563">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="563"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:tax_accountant&quot;</span>, hf_subset=<span class="hljs-string">&quot;tax_accountant&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L564">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="564"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    CustomCEVALEvaluationTask(name=<span class="hljs-string">&quot;ceval:physician&quot;</span>, hf_subset=<span class="hljs-string">&quot;physician&quot;</span>),<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L565">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="565"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->]<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L566">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="566"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L567">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="567"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->ceval_subject_mapping = {<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L568">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="568"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;computer_network&#x27;</span>: [<span class="hljs-string">&#x27;Computer Network&#x27;</span>, <span class="hljs-string">&#x27;计算机网络&#x27;</span>, <span class="hljs-string">&#x27;STEM&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L569">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="569"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;operating_system&#x27;</span>: [<span class="hljs-string">&#x27;Operating System&#x27;</span>, <span class="hljs-string">&#x27;操作系统&#x27;</span>, <span class="hljs-string">&#x27;STEM&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L570">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="570"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;computer_architecture&#x27;</span>: [<span class="hljs-string">&#x27;Computer Architecture&#x27;</span>, <span class="hljs-string">&#x27;计算机组成&#x27;</span>, <span class="hljs-string">&#x27;STEM&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L571">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="571"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;college_programming&#x27;</span>: [<span class="hljs-string">&#x27;College Programming&#x27;</span>, <span class="hljs-string">&#x27;大学编程&#x27;</span>, <span class="hljs-string">&#x27;STEM&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L572">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="572"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;college_physics&#x27;</span>: [<span class="hljs-string">&#x27;College Physics&#x27;</span>, <span class="hljs-string">&#x27;大学物理&#x27;</span>, <span class="hljs-string">&#x27;STEM&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L573">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="573"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;college_chemistry&#x27;</span>: [<span class="hljs-string">&#x27;College Chemistry&#x27;</span>, <span class="hljs-string">&#x27;大学化学&#x27;</span>, <span class="hljs-string">&#x27;STEM&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L574">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="574"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;advanced_mathematics&#x27;</span>: [<span class="hljs-string">&#x27;Advanced Mathematics&#x27;</span>, <span class="hljs-string">&#x27;高等数学&#x27;</span>, <span class="hljs-string">&#x27;STEM&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L575">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="575"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;probability_and_statistics&#x27;</span>: [<span class="hljs-string">&#x27;Probability and Statistics&#x27;</span>, <span class="hljs-string">&#x27;概率统计&#x27;</span>, <span class="hljs-string">&#x27;STEM&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L576">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="576"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;discrete_mathematics&#x27;</span>: [<span class="hljs-string">&#x27;Discrete Mathematics&#x27;</span>, <span class="hljs-string">&#x27;离散数学&#x27;</span>, <span class="hljs-string">&#x27;STEM&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L577">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="577"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;electrical_engineer&#x27;</span>: [<span class="hljs-string">&#x27;Electrical Engineer&#x27;</span>, <span class="hljs-string">&#x27;注册电气工程师&#x27;</span>, <span class="hljs-string">&#x27;STEM&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L578">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="578"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;metrology_engineer&#x27;</span>: [<span class="hljs-string">&#x27;Metrology Engineer&#x27;</span>, <span class="hljs-string">&#x27;注册计量师&#x27;</span>, <span class="hljs-string">&#x27;STEM&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L579">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="579"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;high_school_mathematics&#x27;</span>: [<span class="hljs-string">&#x27;High School Mathematics&#x27;</span>, <span class="hljs-string">&#x27;高中数学&#x27;</span>, <span class="hljs-string">&#x27;STEM&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L580">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="580"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;high_school_physics&#x27;</span>: [<span class="hljs-string">&#x27;High School Physics&#x27;</span>, <span class="hljs-string">&#x27;高中物理&#x27;</span>, <span class="hljs-string">&#x27;STEM&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L581">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="581"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;high_school_chemistry&#x27;</span>: [<span class="hljs-string">&#x27;High School Chemistry&#x27;</span>, <span class="hljs-string">&#x27;高中化学&#x27;</span>, <span class="hljs-string">&#x27;STEM&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L582">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="582"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;high_school_biology&#x27;</span>: [<span class="hljs-string">&#x27;High School Biology&#x27;</span>, <span class="hljs-string">&#x27;高中生物&#x27;</span>, <span class="hljs-string">&#x27;STEM&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L583">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="583"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;middle_school_mathematics&#x27;</span>: [<span class="hljs-string">&#x27;Middle School Mathematics&#x27;</span>, <span class="hljs-string">&#x27;初中数学&#x27;</span>, <span class="hljs-string">&#x27;STEM&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L584">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="584"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;middle_school_biology&#x27;</span>: [<span class="hljs-string">&#x27;Middle School Biology&#x27;</span>, <span class="hljs-string">&#x27;初中生物&#x27;</span>, <span class="hljs-string">&#x27;STEM&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L585">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="585"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;middle_school_physics&#x27;</span>: [<span class="hljs-string">&#x27;Middle School Physics&#x27;</span>, <span class="hljs-string">&#x27;初中物理&#x27;</span>, <span class="hljs-string">&#x27;STEM&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L586">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="586"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;middle_school_chemistry&#x27;</span>: [<span class="hljs-string">&#x27;Middle School Chemistry&#x27;</span>, <span class="hljs-string">&#x27;初中化学&#x27;</span>, <span class="hljs-string">&#x27;STEM&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L587">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="587"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;veterinary_medicine&#x27;</span>: [<span class="hljs-string">&#x27;Veterinary Medicine&#x27;</span>, <span class="hljs-string">&#x27;兽医学&#x27;</span>, <span class="hljs-string">&#x27;STEM&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L588">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="588"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;college_economics&#x27;</span>: [<span class="hljs-string">&#x27;College Economics&#x27;</span>, <span class="hljs-string">&#x27;大学经济学&#x27;</span>, <span class="hljs-string">&#x27;Social Science&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L589">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="589"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;business_administration&#x27;</span>: [<span class="hljs-string">&#x27;Business Administration&#x27;</span>, <span class="hljs-string">&#x27;工商管理&#x27;</span>, <span class="hljs-string">&#x27;Social Science&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L590">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="590"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;marxism&#x27;</span>: [<span class="hljs-string">&#x27;Marxism&#x27;</span>, <span class="hljs-string">&#x27;马克思主义基本原理&#x27;</span>, <span class="hljs-string">&#x27;Social Science&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L591">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="591"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;mao_zedong_thought&#x27;</span>: [<span class="hljs-string">&#x27;Mao Zedong Thought&#x27;</span>, <span class="hljs-string">&#x27;毛泽东思想和中国特色社会主义理论体系概论&#x27;</span>, <span class="hljs-string">&#x27;Social Science&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L592">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="592"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;education_science&#x27;</span>: [<span class="hljs-string">&#x27;Education Science&#x27;</span>, <span class="hljs-string">&#x27;教育学&#x27;</span>, <span class="hljs-string">&#x27;Social Science&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L593">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="593"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;teacher_qualification&#x27;</span>: [<span class="hljs-string">&#x27;Teacher Qualification&#x27;</span>, <span class="hljs-string">&#x27;教师资格&#x27;</span>, <span class="hljs-string">&#x27;Social Science&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L594">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="594"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;high_school_politics&#x27;</span>: [<span class="hljs-string">&#x27;High School Politics&#x27;</span>, <span class="hljs-string">&#x27;高中政治&#x27;</span>, <span class="hljs-string">&#x27;Social Science&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L595">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="595"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;high_school_geography&#x27;</span>: [<span class="hljs-string">&#x27;High School Geography&#x27;</span>, <span class="hljs-string">&#x27;高中地理&#x27;</span>, <span class="hljs-string">&#x27;Social Science&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L596">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="596"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;middle_school_politics&#x27;</span>: [<span class="hljs-string">&#x27;Middle School Politics&#x27;</span>, <span class="hljs-string">&#x27;初中政治&#x27;</span>, <span class="hljs-string">&#x27;Social Science&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L597">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="597"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;middle_school_geography&#x27;</span>: [<span class="hljs-string">&#x27;Middle School Geography&#x27;</span>, <span class="hljs-string">&#x27;初中地理&#x27;</span>, <span class="hljs-string">&#x27;Social Science&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L598">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="598"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;modern_chinese_history&#x27;</span>: [<span class="hljs-string">&#x27;Modern Chinese History&#x27;</span>, <span class="hljs-string">&#x27;近代史纲要&#x27;</span>, <span class="hljs-string">&#x27;Humanities&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L599">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="599"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;ideological_and_moral_cultivation&#x27;</span>: [<span class="hljs-string">&#x27;Ideological and Moral Cultivation&#x27;</span>, <span class="hljs-string">&#x27;思想道德修养与法律基础&#x27;</span>, <span class="hljs-string">&#x27;Humanities&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L600">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="600"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;logic&#x27;</span>: [<span class="hljs-string">&#x27;Logic&#x27;</span>, <span class="hljs-string">&#x27;逻辑学&#x27;</span>, <span class="hljs-string">&#x27;Humanities&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L601">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="601"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;law&#x27;</span>: [<span class="hljs-string">&#x27;Law&#x27;</span>, <span class="hljs-string">&#x27;法学&#x27;</span>, <span class="hljs-string">&#x27;Humanities&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L602">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="602"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;chinese_language_and_literature&#x27;</span>: [<span class="hljs-string">&#x27;Chinese Language and Literature&#x27;</span>, <span class="hljs-string">&#x27;中国语言文学&#x27;</span>, <span class="hljs-string">&#x27;Humanities&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L603">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="603"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;art_studies&#x27;</span>: [<span class="hljs-string">&#x27;Art Studies&#x27;</span>, <span class="hljs-string">&#x27;艺术学&#x27;</span>, <span class="hljs-string">&#x27;Humanities&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L604">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="604"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;professional_tour_guide&#x27;</span>: [<span class="hljs-string">&#x27;Professional Tour Guide&#x27;</span>, <span class="hljs-string">&#x27;导游资格&#x27;</span>, <span class="hljs-string">&#x27;Humanities&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L605">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="605"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;legal_professional&#x27;</span>: [<span class="hljs-string">&#x27;Legal Professional&#x27;</span>, <span class="hljs-string">&#x27;法律职业资格&#x27;</span>, <span class="hljs-string">&#x27;Humanities&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L606">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="606"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;high_school_chinese&#x27;</span>: [<span class="hljs-string">&#x27;High School Chinese&#x27;</span>, <span class="hljs-string">&#x27;高中语文&#x27;</span>, <span class="hljs-string">&#x27;Humanities&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L607">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="607"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;high_school_history&#x27;</span>: [<span class="hljs-string">&#x27;High School History&#x27;</span>, <span class="hljs-string">&#x27;高中历史&#x27;</span>, <span class="hljs-string">&#x27;Humanities&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L608">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="608"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;middle_school_history&#x27;</span>: [<span class="hljs-string">&#x27;Middle School History&#x27;</span>, <span class="hljs-string">&#x27;初中历史&#x27;</span>, <span class="hljs-string">&#x27;Humanities&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L609">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="609"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;civil_servant&#x27;</span>: [<span class="hljs-string">&#x27;Civil Servant&#x27;</span>, <span class="hljs-string">&#x27;公务员&#x27;</span>, <span class="hljs-string">&#x27;Other&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L610">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="610"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;sports_science&#x27;</span>: [<span class="hljs-string">&#x27;Sports Science&#x27;</span>, <span class="hljs-string">&#x27;体育学&#x27;</span>, <span class="hljs-string">&#x27;Other&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L611">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="611"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;plant_protection&#x27;</span>: [<span class="hljs-string">&#x27;Plant Protection&#x27;</span>, <span class="hljs-string">&#x27;植物保护&#x27;</span>, <span class="hljs-string">&#x27;Other&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L612">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="612"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;basic_medicine&#x27;</span>: [<span class="hljs-string">&#x27;Basic Medicine&#x27;</span>, <span class="hljs-string">&#x27;基础医学&#x27;</span>, <span class="hljs-string">&#x27;Other&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L613">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="613"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;clinical_medicine&#x27;</span>: [<span class="hljs-string">&#x27;Clinical Medicine&#x27;</span>, <span class="hljs-string">&#x27;临床医学&#x27;</span>, <span class="hljs-string">&#x27;Other&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L614">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="614"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;urban_and_rural_planner&#x27;</span>: [<span class="hljs-string">&#x27;Urban and Rural Planner&#x27;</span>, <span class="hljs-string">&#x27;注册城乡规划师&#x27;</span>, <span class="hljs-string">&#x27;Other&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L615">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="615"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;accountant&#x27;</span>: [<span class="hljs-string">&#x27;Accountant&#x27;</span>, <span class="hljs-string">&#x27;注册会计师&#x27;</span>, <span class="hljs-string">&#x27;Other&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L616">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="616"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;fire_engineer&#x27;</span>: [<span class="hljs-string">&#x27;Fire Engineer&#x27;</span>, <span class="hljs-string">&#x27;注册消防工程师&#x27;</span>, <span class="hljs-string">&#x27;Other&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L617">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="617"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;environmental_impact_assessment_engineer&#x27;</span>: [<span class="hljs-string">&#x27;Environmental Impact Assessment Engineer&#x27;</span>, <span class="hljs-string">&#x27;环境影响评价工程师&#x27;</span>, <span class="hljs-string">&#x27;Other&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L618">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="618"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;tax_accountant&#x27;</span>: [<span class="hljs-string">&#x27;Tax Accountant&#x27;</span>, <span class="hljs-string">&#x27;税务师&#x27;</span>, <span class="hljs-string">&#x27;Other&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L619">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="619"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&#x27;physician&#x27;</span>: [<span class="hljs-string">&#x27;Physician&#x27;</span>, <span class="hljs-string">&#x27;医师资格&#x27;</span>, <span class="hljs-string">&#x27;Other&#x27;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L620">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="620"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->}<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L621">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="621"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L622">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="622"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-keyword">def</span> <span class="hljs-title function_">ceval_prompt</span>(<span class="hljs-params">line, task_name: <span class="hljs-built_in">str</span> = <span class="hljs-literal">None</span></span>):<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L623">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="623"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-comment"># f&quot;以下是中国关于{_ch_name}考试的单项选择题，请选出其中的正确答案。\n{{question}}\nA. {{A}}\nB. {{B}}\nC. {{C}}\nD. {{D}}\n答案: &quot;</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L624">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="624"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&quot;&quot;&quot;CEVAL prompt without letters&quot;&quot;&quot;</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L625">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="625"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    topic = ceval_subject_mapping[line[<span class="hljs-string">&#x27;subject&#x27;</span>]][<span class="hljs-number">1</span>]<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L626">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="626"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    prompt = <span class="hljs-string">f&quot;以下是中国关于<span class="hljs-subst">{topic.replace(<span class="hljs-string">&#x27;_&#x27;</span>, <span class="hljs-string">&#x27; &#x27;</span>)}</span>考试的单项选择题，请选出其中的正确答案。\n题目：&quot;</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L627">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="627"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    prompt += line[<span class="hljs-string">&quot;question&quot;</span>] + <span class="hljs-string">&quot;\n答案：&quot;</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L628">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="628"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-comment">#print(f&quot;ceval_prompt={prompt}&quot;)</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L629">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="629"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L630">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="630"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-keyword">return</span> Doc(<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L631">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="631"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        task_name=task_name,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L632">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="632"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        query=prompt,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L633">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="633"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        choices=[<span class="hljs-string">f&quot; <span class="hljs-subst">{c}</span>&quot;</span> <span class="hljs-keyword">for</span> c <span class="hljs-keyword">in</span> line[<span class="hljs-string">&quot;choices&quot;</span>]],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L634">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="634"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        gold_index=line[<span class="hljs-string">&quot;answer&quot;</span>],<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L635">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="635"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->        instruction=<span class="hljs-literal">None</span>,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L636">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="636"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    )<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L637">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="637"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L638">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="638"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->CEVAL_STRING = [(t, <span class="hljs-string">f&quot;custom|<span class="hljs-subst">{t.name}</span>|0|1&quot;</span>) <span class="hljs-keyword">for</span> t <span class="hljs-keyword">in</span> CEVAL_TASKS]<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L639">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="639"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->_TASKS_STRINGS.extend(CEVAL_STRING)<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L640">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="640"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->_TASKS += CEVAL_TASKS<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L641">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="641"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-built_in">print</span>(<span class="hljs-string">f&#x27;<span class="hljs-subst">{<span class="hljs-string">&quot;,&quot;</span>.join([t[<span class="hljs-number">1</span>] <span class="hljs-keyword">for</span> t <span class="hljs-keyword">in</span> CEVAL_STRING])}</span>&#x27;</span>)<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L642">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="642"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L643">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="643"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-comment">############################################################################################################################################################</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L644">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="644"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L645">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="645"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-comment"># common sense reasoning + mmlu</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L646">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="646"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->EARLY_SIGNAL_TASKS = <span class="hljs-string">&quot;,&quot;</span>.join([t[<span class="hljs-number">1</span>] <span class="hljs-keyword">for</span> t <span class="hljs-keyword">in</span> COMMON_SENSE_REASONING_STRING] + [t[<span class="hljs-number">1</span>] <span class="hljs-keyword">for</span> t <span class="hljs-keyword">in</span> MMLU_STRING] + [t[<span class="hljs-number">1</span>] <span class="hljs-keyword">for</span> t <span class="hljs-keyword">in</span> CMMLU_STRING])<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L647">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="647"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L648">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="648"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-comment"># Convert to dict for lighteval</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L649">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="649"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->TASKS_TABLE = [task.as_dict() <span class="hljs-keyword">for</span> task <span class="hljs-keyword">in</span> _TASKS]<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L650">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="650"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START --><span class="hljs-comment"># You can have a few pre-organised groups of tasks</span><!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L651">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="651"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->TASKS_GROUPS = {<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L652">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="652"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->    <span class="hljs-string">&quot;early-signal&quot;</span>: EARLY_SIGNAL_TASKS,<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L653">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="653"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->}<!-- HTML_TAG_END --></td>
					</tr><tr class="" id="L654">
						
						<td class="blob-line-num w-1 cursor-pointer select-none pl-5 pr-3 text-right align-top text-gray-300 hover:text-black dark:hover:text-white" data-line-num="654"></td>
						<td class="blob-line overflow-visible whitespace-pre px-3"><!-- HTML_TAG_START -->
<!-- HTML_TAG_END --></td>
					</tr></tbody></table></div>
	</div></div></div></div></section></div></main>

	</div>

		<script>
			import("/front/build/kube-47f8e3c/index.js");
			window.moonSha = "kube-47f8e3c/";
		</script>

		<!-- Stripe -->
		<script>
			if (["hf.co", "huggingface.co"].includes(window.location.hostname)) {
				const script = document.createElement("script");
				script.src = "https://js.stripe.com/v3/";
				script.async = true;
				document.head.appendChild(script);
			}
		</script>
	</body>
</html>
