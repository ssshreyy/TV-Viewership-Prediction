import { BrowserModule } from '@angular/platform-browser';
import { FormsModule } from '@angular/forms';
import { NgModule } from '@angular/core';
import { PagesModule } from './pages/pages.module';
import { routing } from './app.routing';
import { AppComponent } from './app.component';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { PreprocessModule } from './pages/preprocess/preprocess.module';
import { VisualsModule } from './pages/visuals/visuals.module';
import { SentimentModule } from './pages/sentiment/sentiment.module';

@NgModule({
  imports: [
    BrowserModule,
    BrowserAnimationsModule,
    FormsModule,
    PreprocessModule,
    VisualsModule,
    PagesModule,
    routing,
    SentimentModule
  ],
  declarations: [
    AppComponent,
  ],
  bootstrap: [AppComponent]
})
export class AppModule { }
