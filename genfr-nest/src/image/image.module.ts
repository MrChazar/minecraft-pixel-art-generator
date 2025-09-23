import { Module } from '@nestjs/common';
import { ImageService } from './image.service';
import { ImageController } from './image.controller';
import { PrismaModule } from '../prisma/prisma.module';
import {   HttpModule } from '@nestjs/axios';
import { AuthModule } from '../auth/auth.module';

@Module({
  controllers: [ImageController],
  providers: [ImageService],
  imports:[PrismaModule,HttpModule,AuthModule],
  exports:[ImageService]
})
export class ImageModule {}
